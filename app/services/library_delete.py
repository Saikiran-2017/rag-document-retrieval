"""Remove a document from the raw library and refresh the FAISS index."""

from __future__ import annotations

from pathlib import Path

from app.persistence import document_manifest, index_library_state
from app.retrieval.vector_store import DEFAULT_INDEX_NAME
from app.services import index_service
from app.services.upload_service import sanitize_upload_basename


def _remove_faiss_artifacts(faiss_folder: Path) -> None:
    folder = Path(faiss_folder).resolve()
    for ext in (".faiss", ".pkl"):
        p = folder / f"{DEFAULT_INDEX_NAME}{ext}"
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass
    sp = index_library_state.state_path(folder)
    try:
        if sp.is_file():
            sp.unlink()
    except OSError:
        pass


def delete_library_document(
    raw_dir: Path,
    faiss_folder: Path,
    filename: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[bool, str]:
    """
    Delete one supported file under ``raw_dir`` and rebuild the index (or clear artifacts if empty).

    Returns ``(ok, message)`` where ``message`` is user-facing on failure.
    """
    safe, reason = sanitize_upload_basename(filename)
    if safe is None:
        return False, reason or "invalid_filename"

    raw_resolved = raw_dir.resolve()
    path = (raw_resolved / safe).resolve()
    try:
        path.relative_to(raw_resolved)
    except ValueError:
        return False, "invalid_path"
    if not path.is_file():
        return False, "not_found"

    try:
        path.unlink()
    except OSError as exc:
        return False, str(exc)

    remaining = index_service.list_raw_files(raw_dir)
    if not remaining:
        _remove_faiss_artifacts(faiss_folder)
        document_manifest.clear_all_file_rows(faiss_folder)
        return True, "Removed file and cleared index."

    ok, msg, _n, _action = index_service.rebuild_knowledge_index(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return ok, msg if ok else msg
