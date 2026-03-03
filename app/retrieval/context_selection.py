"""
Rerank and trim hybrid retrieval hits before prompt construction.

Improves signal-to-noise: fusion score + vector distance composite, weak-chunk filtering,
then a bounded context size for generation (sources in UI match CONTEXT blocks).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Literal

from app.retrieval.vector_store import RetrievedChunk

TaskMode = Literal["qa", "summarize", "extract", "compare"]


def composite_retrieval_score(hit: RetrievedChunk) -> float:
    """
    Higher = better. Combines RRF mass with inverse vector distance.

    BM25-only hits use a high sentinel distance in hybrid_retrieve; RRF still moves them
    when keyword match is strong.
    """
    rrf = float(hit.metadata.get("rrf_score") or 0.0)
    d = max(float(hit.distance), 1e-6)
    inv_d = 1.0 / (1.0 + d)
    return 3.0 * rrf + inv_d


def rerank_hybrid_hits(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Stable rerank by composite score (desc), then original rank, then distance."""
    scored = sorted(
        enumerate(hits),
        key=lambda t: (-composite_retrieval_score(t[1]), t[1].rank, float(t[1].distance)),
    )
    out: list[RetrievedChunk] = []
    for new_r, (_, h) in enumerate(scored):
        meta = dict(h.metadata)
        meta["rerank_score"] = round(composite_retrieval_score(h), 6)
        meta["rerank_order"] = new_r
        out.append(
            RetrievedChunk(
                rank=new_r,
                page_content=h.page_content,
                metadata=meta,
                distance=float(h.distance),
                score_kind=h.score_kind,
            )
        )
    return out


def _is_weak_chunk(hit: RetrievedChunk) -> bool:
    """Drop obvious noise from the generation pool (not used for top-hit gate)."""
    d = float(hit.distance)
    rrf = float(hit.metadata.get("rrf_score") or 0.0)
    if d >= 1.45 and rrf < 0.0095:
        return True
    if d >= 1.62:
        return True
    return False


def filter_generation_candidates(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove weak tail chunks; keep order."""
    return [h for h in hits if not _is_weak_chunk(h)]


def diversify_chunks_round_robin(candidates: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    """
    Interleave chunks across ``source_name`` so multi-file libraries surface in broad Q&A.

    Preserves per-file order from ``candidates``; only used when multiple sources exist.
    """
    if limit < 1 or not candidates:
        return []
    by_src: dict[str, deque[RetrievedChunk]] = defaultdict(deque)
    order: list[str] = []
    for h in candidates:
        sn = str(h.metadata.get("source_name") or "").strip() or "_unknown"
        if sn not in by_src:
            order.append(sn)
        by_src[sn].append(h)
    if len(order) <= 1:
        return list(candidates)[:limit]
    out: list[RetrievedChunk] = []
    while len(out) < limit and any(by_src.values()):
        for sn in order:
            q = by_src.get(sn)
            if q and len(out) < limit:
                out.append(q.popleft())
    return out


def context_limit_for_task(
    mode: TaskMode, top_k: int, nvec: int, *, broad_document_question: bool = False
) -> int:
    """How many chunks to pass into the LLM after rerank/filter."""
    tk = max(1, int(top_k))
    if mode == "qa":
        upper = 10 if broad_document_question else 5
        k = min(max(tk, 3), upper)
    elif mode == "summarize":
        k = min(nvec, max(8, tk * 2))
    elif mode == "compare":
        k = min(nvec, max(10, tk * 2))
    else:
        k = min(nvec, max(5, tk))
    return max(1, min(k, nvec))


def select_generation_context(
    ranked_hits: list[RetrievedChunk],
    *,
    mode: TaskMode,
    top_k: int,
    nvec: int,
    broad_document_question: bool = False,
) -> list[RetrievedChunk]:
    """
    Reranked list → filter weak → take top ``context_limit_for_task``.

    Returned chunks are exactly those numbered [SOURCE 1..N] in the prompt.
    """
    filtered = filter_generation_candidates(ranked_hits)
    if not filtered:
        filtered = ranked_hits[:1] if ranked_hits else []
    limit = context_limit_for_task(
        mode, top_k, nvec, broad_document_question=broad_document_question
    )
    if broad_document_question and mode == "qa":
        names = {str(h.metadata.get("source_name") or "").strip() for h in filtered}
        if len({n for n in names if n}) > 1:
            return diversify_chunks_round_robin(filtered, limit)
    return filtered[:limit]


def hybrid_pool_size(nvec: int, top_k: int) -> int:
    """Candidate pool size for RRF before rerank (wider than final context)."""
    return min(nvec, max(14, int(top_k) * 3, 18))
