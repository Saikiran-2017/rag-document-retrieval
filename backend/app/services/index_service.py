"""Re-exports indexing / sync (FAISS, manifest, incremental builds)."""

from app.services.index_service import (
    ensure_index_matches_library,
    library_content_fingerprint,
    list_raw_files,
    rebuild_knowledge_index,
)

__all__ = [
    "ensure_index_matches_library",
    "library_content_fingerprint",
    "list_raw_files",
    "rebuild_knowledge_index",
]
