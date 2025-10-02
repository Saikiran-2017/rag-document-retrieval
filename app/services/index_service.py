"""Library listing, fingerprinting, FAISS build/load, and sync with session state."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.documents import Document

from app.ingestion.loader import load_file
from app.persistence import document_manifest, index_library_state
from app.retrieval.embedding_cache import cache_dir_for, embed_texts_with_cache
from app.retrieval.index_probe import run_retrieval_self_probe
from app.retrieval.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    build_faiss_from_documents_with_embeddings,
    chunks_to_documents,
    create_openai_embeddings,
    faiss_index_files_exist,
    faiss_vector_count,
    iter_faiss_documents,
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

logger = logging.getLogger(__name__)


def _streamlit_script_running() -> bool:
    """True only inside an active Streamlit script (avoid touching session_state from FastAPI / tests)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _count_chunks_by_source(store: Any) -> dict[str, int]:
    c: Counter[str] = Counter()
    for d in iter_faiss_documents(store):
        sn = str(d.metadata.get("source_name") or "")
        if sn:
            c[sn] += 1
    return dict(c)


def list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        return []
    return sorted(
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def _fmt_content_fp_short(fp: tuple[tuple[str, str], ...]) -> str:
    if not fp:
        return "(empty)"
    parts = [f"{n}" for n, _ in fp[:24]]
    tail = "..." if len(fp) > 24 else ""
    return ", ".join(parts) + tail


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def library_content_fingerprint(raw_dir: Path) -> tuple[tuple[str, str], ...]:
    """Stable fingerprint: (filename, sha256 of file bytes), sorted by name."""
    files = list_raw_files(raw_dir)
    return tuple((p.name, sha256_file(p)) for p in files)


def library_fingerprint(raw_dir: Path) -> tuple[tuple[str, int], ...]:
    """Legacy mtime-based fingerprint (e.g. quick display). Prefer :func:`library_content_fingerprint` for sync."""
    files = list_raw_files(raw_dir)
    return tuple((p.name, int(p.stat().st_mtime_ns)) for p in files)


@st.cache_resource(show_spinner=False)
def cached_openai_embeddings(model: str) -> Any:
    """Reuse one OpenAI embeddings client per model (safe; no stale index risk)."""
    return create_openai_embeddings(model=model)


def load_faiss_store(faiss_folder: Path) -> Any:
    """Load FAISS from disk each time. Avoids stale in-memory index after rebuilds."""
    return load_faiss_index(
        folder_path=faiss_folder,
        index_name=DEFAULT_INDEX_NAME,
        embeddings=cached_openai_embeddings(DEFAULT_EMBEDDING_MODEL),
    )


def _settings_match_saved_state(
    state: dict[str, Any] | None,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> bool:
    if not state:
        return False
    return (
        int(state.get("chunk_size", -1)) == chunk_size
        and int(state.get("chunk_overlap", -1)) == chunk_overlap
        and str(state.get("embedding_model", "")) == embedding_model
    )


def rebuild_knowledge_index(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[bool, str, int, str]:
    """
    Returns ``(ok, message, vector_count, sync_action)`` where ``sync_action`` is
    ``unchanged`` (hashes + settings match; incremental skip), ``rebuilt``, or ``failed``.
    """
    if chunk_overlap >= chunk_size:
        document_manifest.mark_index_failure(faiss_folder, MSG_PREPARE_SETTINGS_HINT)
        return False, MSG_PREPARE_SETTINGS_HINT, 0, "failed"

    model = DEFAULT_EMBEDDING_MODEL
    embeddings = cached_openai_embeddings(model)
    cache_root = cache_dir_for(faiss_folder, model)

    try:
        content_fp = library_content_fingerprint(raw_dir)
        if not content_fp:
            document_manifest.mark_index_failure(faiss_folder, MSG_PREPARE_DOCS_FAILED)
            return False, MSG_PREPARE_DOCS_FAILED, 0, "failed"

        current_hashes = dict(content_fp)
        state = index_library_state.load_state(faiss_folder)
        index_ready = faiss_index_files_exist(faiss_folder, index_name=DEFAULT_INDEX_NAME)

        prev_files: dict[str, str] = dict(state.get("files") or {}) if state else {}
        if (
            index_ready
            and _settings_match_saved_state(state, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embedding_model=model)
            and set(prev_files.keys()) == set(current_hashes.keys())
            and all(prev_files.get(n) == h for n, h in current_hashes.items())
        ):
            store = load_faiss_index(
                folder_path=faiss_folder,
                index_name=DEFAULT_INDEX_NAME,
                embeddings=embeddings,
            )
            nvec = faiss_vector_count(store)
            probe = run_retrieval_self_probe(store)
            actual_counts = _count_chunks_by_source(store)
            file_chunk_counts: dict[str, int] = dict(state.get("file_chunk_counts") or {})
            document_manifest.apply_probe_only_refresh(
                faiss_folder,
                filenames=list(current_hashes.keys()),
                file_chunk_counts=file_chunk_counts,
                actual_chunk_counts=actual_counts,
                retrieval_probe=probe,
            )
            if debug_service.debug_enabled():
                debug_service.merge(index_sync="incremental_skip_unchanged", index_vector_count=nvec)
            logger.info("Index unchanged (file hashes + settings match); skipped rebuild (%s vectors).", nvec)
            return True, "", nvec, "unchanged"

        document_manifest.set_all_processing(faiss_folder, list(current_hashes.keys()))

        old_by_source: dict[str, list[Document]] = defaultdict(list)
        if index_ready:
            try:
                old_store = load_faiss_index(
                    folder_path=faiss_folder,
                    index_name=DEFAULT_INDEX_NAME,
                    embeddings=embeddings,
                )
                for d in iter_faiss_documents(old_store):
                    sn = str(d.metadata.get("source_name") or "")
                    old_by_source[sn].append(d)
                for docs in old_by_source.values():
                    docs.sort(key=lambda x: str(x.metadata.get("chunk_id") or ""))
            except Exception as exc:
                logger.warning("Could not load existing FAISS for incremental merge: %s", exc)
                old_by_source.clear()

        unchanged_names: set[str] = set()
        if state and _settings_match_saved_state(
            state, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embedding_model=model
        ):
            for name, h in current_hashes.items():
                if prev_files.get(name) == h and name in old_by_source:
                    unchanged_names.add(name)

        file_build: dict[str, dict[str, Any]] = {}
        final_documents: list[Document] = []
        for name in sorted(current_hashes.keys()):
            path = raw_dir / name
            if name in unchanged_names:
                reused = old_by_source.get(name, [])
                file_build[name] = {"ok": True, "expected_chunks": len(reused), "reused": True}
                final_documents.extend(reused)
                continue
            try:
                ingested = load_file(path)
            except Exception as exc:
                logger.warning("Skipping %s during index: %s", name, exc)
                file_build[name] = {
                    "ok": False,
                    "reason": "parse",
                    "expected_chunks": 0,
                    "internal": str(exc),
                }
                continue
            if not ingested:
                file_build[name] = {"ok": False, "reason": "empty_ingest", "expected_chunks": 0}
                continue
            chunks = chunk_ingested_documents(
                ingested,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if not chunks:
                file_build[name] = {"ok": False, "reason": "no_chunks", "expected_chunks": 0}
                continue
            file_build[name] = {"ok": True, "expected_chunks": len(chunks), "reused": False}
            final_documents.extend(chunks_to_documents(chunks))

        if not final_documents:
            document_manifest.apply_post_index_validation(
                faiss_folder,
                file_build=file_build,
                actual_chunk_counts={},
                retrieval_probe={"ok": False},
            )
            return False, MSG_READ_FILES_FAILED, 0, "failed"

        texts = [d.page_content for d in final_documents]
        vectors = embed_texts_with_cache(texts, embeddings, cache_root)
        store = build_faiss_from_documents_with_embeddings(final_documents, vectors, embedder=embeddings)
        save_faiss_index(store, folder_path=faiss_folder, index_name=DEFAULT_INDEX_NAME)
        nvec = faiss_vector_count(store)

        actual_counts = _count_chunks_by_source(store)
        probe = run_retrieval_self_probe(store)
        document_manifest.apply_post_index_validation(
            faiss_folder,
            file_build=file_build,
            actual_chunk_counts=actual_counts,
            retrieval_probe=probe,
        )
        file_chunk_counts = {name: int(fb.get("expected_chunks") or 0) for name, fb in file_build.items()}
        index_library_state.save_state(
            faiss_folder,
            {
                "embedding_model": model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "files": dict(current_hashes),
                "vector_count": nvec,
                "file_chunk_counts": file_chunk_counts,
            },
        )
        changed_ct = len(current_hashes) - len(unchanged_names)
        logger.info(
            "Built FAISS: %s vectors (%s file(s) rechunked, %s reused from index).",
            nvec,
            changed_ct,
            len(unchanged_names),
        )
        if debug_service.debug_enabled():
            debug_service.merge(
                index_sync="incremental_rebuilt",
                index_vector_count=nvec,
                index_files_rechunked=changed_ct,
                index_files_reused=len(unchanged_names),
            )
        return True, "", nvec, "rebuilt"
    except Exception as exc:
        document_manifest.mark_index_failure(faiss_folder, str(exc))
        logger.exception("Index rebuild failed")
        return False, MSG_PREPARE_DOCS_FAILED, 0, "failed"


def ensure_index_matches_library(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[bool, str]:
    fp = library_content_fingerprint(raw_dir)
    if not fp:
        if debug_service.debug_enabled():
            debug_service.merge(library_content_fingerprint="(empty)", index_sync="none_no_files")
        return False, MSG_NO_DOCS

    synced = (
        st.session_state.get("kb_sync_fingerprint") if _streamlit_script_running() else None
    )
    index_ready = faiss_index_files_exist(faiss_folder, index_name=DEFAULT_INDEX_NAME)

    if debug_service.debug_enabled():
        debug_service.merge(
            library_content_fingerprint=_fmt_content_fp_short(fp),
            kb_sync_matches=(synced == fp),
        )

    if index_ready and synced is not None and synced == fp:
        if debug_service.debug_enabled():
            debug_service.merge(index_sync="reused")
        return True, ""

    ok, msg, _, _action = rebuild_knowledge_index(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if ok and _streamlit_script_running():
        st.session_state.kb_sync_fingerprint = fp
        if debug_service.debug_enabled():
            debug_service.merge(index_sync="rebuilt")
        return True, ""
    if debug_service.debug_enabled():
        debug_service.merge(index_sync="rebuild_failed", ensure_error_summary=str(msg)[:200])
    return False, msg
