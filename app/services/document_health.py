"""Trust routing: when document-grounded answers are allowed given manifest health + retrieval strength."""

from __future__ import annotations

from pathlib import Path

from app.llm.generator import hybrid_hit_strong_for_limited_corpora, hybrid_retrieval_is_useful
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


def allow_document_grounding(
    faiss_folder: Path,
    hits: list[RetrievedChunk],
    *,
    for_document_task: bool = False,
) -> bool:
    """
    Combine hybrid usefulness (Phase 12) with Phase 13 file health.

    Failed/processing sources are excluded. ready_limited (or a library with no fully
    ready file) requires a stronger top hit before grounding.
    """
    if not hybrid_retrieval_is_useful(hits, for_document_task=for_document_task):
        return False
    top_src = str(hits[0].metadata.get("source_name") or "").strip()
    if not top_src:
        if document_manifest.library_has_no_fully_healthy_file(faiss_folder):
            return hybrid_hit_strong_for_limited_corpora(hits[0], for_document_task=for_document_task)
        return True

    st = document_manifest.file_health_status(faiss_folder, top_src)
    if st == "ready":
        return True
    if st in ("failed", "processing"):
        return False
    if st == "ready_limited":
        return hybrid_hit_strong_for_limited_corpora(hits[0], for_document_task=for_document_task)
    if document_manifest.library_has_no_fully_healthy_file(faiss_folder):
        return hybrid_hit_strong_for_limited_corpora(hits[0], for_document_task=for_document_task)
    return True
