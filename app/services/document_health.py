"""Trust routing: when document-grounded answers are allowed given manifest health + retrieval strength."""

from __future__ import annotations

from pathlib import Path

from app.llm.generator import (
    hybrid_hit_strong_for_limited_corpora,
    hybrid_limited_same_source_fallback,
    hybrid_retrieval_is_useful,
)
from app.persistence import document_manifest
from app.retrieval.vector_store import RetrievedChunk


def filter_trusted_retrieval_hits(
    faiss_folder: Path,
    hits: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Drop hits from files that are not safe to cite (failed or still indexing)."""
    out: list[RetrievedChunk] = []
    for h in hits:
        sn = str(h.metadata.get("source_name") or "").strip()
        if not sn:
            out.append(h)
            continue
        st = document_manifest.file_health_status(faiss_folder, sn)
        if st in ("failed", "processing"):
            continue
        out.append(h)
    return out


def _count_source_in_window(hits: list[RetrievedChunk], source_name: str, window: int = 8) -> int:
    if not source_name:
        return 0
    return sum(
        1
        for h in hits[:window]
        if str(h.metadata.get("source_name") or "").strip() == source_name
    )


def explain_allow_document_grounding(
    faiss_folder: Path,
    hits: list[RetrievedChunk],
    *,
    for_document_task: bool = False,
    relaxed_doc_qa: bool = False,
    lookup_qa_relaxed: bool = False,
) -> tuple[bool, str]:
    """
    Same decision as :func:`allow_document_grounding` with a short reason for logs / eval.

    ``relaxed_doc_qa`` enables broader hybrid + limited-corpora thresholds for overview /
    summary-style questions and vague performance-on-doc queries (see query_intent).
    ``lookup_qa_relaxed`` slightly loosens hybrid thresholds for sparse entity / role lookups
    (CFO, named person) without using the full broad-overview band.
    """
    if not hits:
        return False, "no_hits_after_trust_filter"

    if hybrid_retrieval_is_useful(hits, for_document_task=for_document_task, for_broad_qa=False):
        pass
    elif relaxed_doc_qa and hybrid_retrieval_is_useful(
        hits, for_document_task=False, for_broad_qa=True
    ):
        pass
    elif lookup_qa_relaxed and hybrid_retrieval_is_useful(
        hits, for_document_task=False, for_lookup_qa=True
    ):
        pass
    else:
        h0 = hits[0]
        d = float(h0.distance)
        rrf = float(h0.metadata.get("rrf_score", 0.0) or 0.0)
        return False, f"hybrid_not_useful top_dist={d:.4f} top_rrf={rrf:.5f}"

    top_src = str(hits[0].metadata.get("source_name") or "").strip()
    n_same = _count_source_in_window(hits, top_src, 8)

    def _limited_ok(*, for_broad: bool = False, for_lookup: bool = False) -> bool:
        return hybrid_hit_strong_for_limited_corpora(
            hits[0],
            for_document_task=for_document_task,
            for_broad_qa=for_broad,
            for_lookup_qa=for_lookup,
            same_source_hits_in_window=n_same,
        )

    def _limited_chain() -> bool:
        ok = _limited_ok()
        if not ok and relaxed_doc_qa:
            ok = _limited_ok(for_broad=True)
        if not ok and lookup_qa_relaxed:
            ok = _limited_ok(for_lookup=True)
        return ok

    if not top_src:
        if document_manifest.library_has_no_fully_healthy_file(faiss_folder):
            ok = _limited_chain()
            if not ok and (relaxed_doc_qa or lookup_qa_relaxed):
                ok = hybrid_limited_same_source_fallback(hits[0], same_source_hits_in_window=n_same)
            return ok, f"no_source_name limited_library strong_enough={ok}"
        return True, "no_source_name_manifest_ok"

    st = document_manifest.file_health_status(faiss_folder, top_src)
    if st == "ready":
        return True, f"manifest_ready source={top_src!r}"
    if st in ("failed", "processing"):
        return False, f"manifest_blocked status={st} source={top_src!r}"
    if st == "ready_limited":
        ok = _limited_chain()
        if not ok and (relaxed_doc_qa or lookup_qa_relaxed):
            ok = hybrid_limited_same_source_fallback(hits[0], same_source_hits_in_window=n_same)
        return ok, f"ready_limited source={top_src!r} strong_enough={ok} same_src_top8={n_same}"
    if document_manifest.library_has_no_fully_healthy_file(faiss_folder):
        ok = _limited_chain()
        if not ok and (relaxed_doc_qa or lookup_qa_relaxed):
            ok = hybrid_limited_same_source_fallback(hits[0], same_source_hits_in_window=n_same)
        return ok, f"no_ready_file strong_enough={ok} same_src_top8={n_same}"
    return True, f"manifest_allow source={top_src!r} status={st!r}"


def allow_document_grounding(
    faiss_folder: Path,
    hits: list[RetrievedChunk],
    *,
    for_document_task: bool = False,
    relaxed_doc_qa: bool = False,
    lookup_qa_relaxed: bool = False,
) -> bool:
    """
    Combine hybrid usefulness (Phase 12) with Phase 13 file health.

    Failed/processing sources are excluded. ready_limited (or a library with no fully
    ready file) requires a stronger top hit before grounding.
    """
    ok, _ = explain_allow_document_grounding(
        faiss_folder,
        hits,
        for_document_task=for_document_task,
        relaxed_doc_qa=relaxed_doc_qa,
        lookup_qa_relaxed=lookup_qa_relaxed,
    )
    return ok
