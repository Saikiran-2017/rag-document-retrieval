"""Library listing, fingerprinting, FAISS build/load, and sync with session state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app.ingestion.loader import load_raw_directory
from app.retrieval.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    build_faiss_from_chunks,
    create_openai_embeddings,
    faiss_index_files_exist,
    faiss_vector_count,
    load_faiss_index,
    save_faiss_index,
)
from app.services import debug_service
from app.services.message_service import (
    MSG_NO_DOCS,
    MSG_PREPARE_DOCS_FAILED,
    MSG_PREPARE_SETTINGS_HINT,
    MSG_READ_FILES_FAILED,
)
from app.services.upload_service import SUPPORTED_EXT
from app.utils.chunker import chunk_ingested_documents


def _fmt_fp_short(fp: tuple[tuple[str, int], ...]) -> str:
    if not fp:
        return "(empty)"
    parts = [f"{n}" for n, _ in fp[:24]]
    tail = "..." if len(fp) > 24 else ""
    return ", ".join(parts) + tail


@st.cache_resource(show_spinner=False)
def cached_openai_embeddings(model: str) -> Any:
    """Reuse one OpenAI embeddings client per model (safe; no stale index risk)."""
    return create_openai_embeddings(model=model)


def list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        return []
    return sorted(
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def library_fingerprint(raw_dir: Path) -> tuple[tuple[str, int], ...]:
    files = list_raw_files(raw_dir)
    return tuple((p.name, int(p.stat().st_mtime_ns)) for p in files)


def load_faiss_store(faiss_folder: Path) -> Any:
    """Load FAISS from disk each time. Avoids stale in-memory index after rebuilds."""
    return load_faiss_index(
        folder_path=faiss_folder,
        index_name=DEFAULT_INDEX_NAME,
        embeddings=cached_openai_embeddings(DEFAULT_EMBEDDING_MODEL),
    )


def rebuild_knowledge_index(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[bool, str, int]:
    if chunk_overlap >= chunk_size:
        return False, MSG_PREPARE_SETTINGS_HINT, 0

    try:
        docs = load_raw_directory(raw_dir)
        if not docs:
            return False, MSG_PREPARE_DOCS_FAILED, 0

        chunks = chunk_ingested_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            return False, MSG_READ_FILES_FAILED, 0

        embeddings = cached_openai_embeddings(DEFAULT_EMBEDDING_MODEL)
        store = build_faiss_from_chunks(chunks, embeddings=embeddings)
        save_faiss_index(store, folder_path=faiss_folder, index_name=DEFAULT_INDEX_NAME)
        nvec = faiss_vector_count(store)
        return True, "", nvec
    except Exception:
        return False, MSG_PREPARE_DOCS_FAILED, 0


def ensure_index_matches_library(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[bool, str]:
    fp = library_fingerprint(raw_dir)
    if not fp:
        if debug_service.debug_enabled():
            debug_service.merge(library_fingerprint="(empty)", index_sync="none_no_files")
        return False, MSG_NO_DOCS

    synced = st.session_state.get("kb_sync_fingerprint")
    index_ready = faiss_index_files_exist(faiss_folder, index_name=DEFAULT_INDEX_NAME)

    if debug_service.debug_enabled():
        debug_service.merge(
            library_fingerprint=_fmt_fp_short(fp),
            kb_sync_matches=(synced == fp),
        )

    if index_ready and synced == fp:
        if debug_service.debug_enabled():
            debug_service.merge(index_sync="reused")
        return True, ""

    ok, msg, _ = rebuild_knowledge_index(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if ok:
        st.session_state.kb_sync_fingerprint = fp
        if debug_service.debug_enabled():
            debug_service.merge(index_sync="rebuilt")
        return True, ""
    if debug_service.debug_enabled():
        debug_service.merge(index_sync="rebuild_failed", ensure_error_summary=str(msg)[:200])
    return False, msg
