"""Post-index retrieval sanity check (index-level, not per-file)."""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS

from app.retrieval.vector_store import faiss_vector_count, retrieve_top_k

# If the best hit to broad self-queries is worse than this, search quality is suspect.
_MAX_ACCEPTABLE_PROBE_L2 = 1.48


def run_retrieval_self_probe(store: FAISS) -> dict[str, Any]:
    """
    Run short probe queries; return ``ok``, best L2, and a short message.

    Used after a successful build to flag indexes that embed poorly (e.g. empty-ish text).
    """
    n = faiss_vector_count(store)
    if n <= 0:
        return {"ok": False, "best_l2": None, "message": "empty_index"}
    if n < 3:
        return {"ok": True, "best_l2": None, "message": "tiny_index_skipped"}

    probes = (
        "summary overview introduction key topics main points",
        "document text content sections paragraphs",
    )
    best: float | None = None
    for q in probes:
        try:
            hits = retrieve_top_k(store, q, k=min(3, n))
        except Exception:
            continue
        if hits:
            d0 = float(hits[0].distance)
            best = d0 if best is None else min(best, d0)

    if best is None:
        return {"ok": False, "best_l2": None, "message": "probe_failed"}

    ok = best <= _MAX_ACCEPTABLE_PROBE_L2
    return {
        "ok": ok,
        "best_l2": round(best, 6),
        "message": "ok" if ok else "weak_best_hit",
    }
