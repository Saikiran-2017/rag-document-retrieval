"""Query routing, grounded vs general answers, safe fallbacks, and assistant message shaping."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from app.llm.generator import (
    GroundedAnswer,
    generate_general_answer,
    generate_grounded_answer,
    retrieval_is_useful,
)
from app.retrieval.vector_store import RetrievedChunk, faiss_vector_count, retrieve_top_k
from app.services import debug_service, index_service
from app.services.message_service import (
    MSG_EMPTY_MESSAGE,
    MSG_GROUNDED_FALLBACK_NOTE,
    MSG_LIBRARY_UNAVAILABLE,
    MSG_SERVICE_UNAVAILABLE,
    merge_notes,
    preview_text,
)


@dataclass
class AssistantTurn:
    """Result of routing: document-grounded vs general assistant vs error."""

    mode: Literal["grounded", "general", "error"]
    text: str
    grounded: GroundedAnswer | None = None
    hits: list[RetrievedChunk] | None = None
    error: str | None = None
    # Muted status line under the answer (library/sync/grounded-fallback hints).
    assistant_note: str | None = None


# If the query clearly does not ask about documents, skip FAISS load + retrieval (same UX).
_DOC_QUERY_HINT = re.compile(
    r"\b(document|documents|file|files|pdf|upload|uploaded|page|pages|passage|passages|"
    r"excerpt|excerpts|cite|citation|source\b|sources\b|section|library|chunk|"
    r"these files|my files|the text|summarize|summary|key points|themes|outline|"
    r"extract|according to|from the|in the document)\b",
    re.I,
)


def wants_no_retrieval_fastpath(query: str) -> bool:
    q = query.strip()
    if len(q) < 4 or len(q) > 220:
        return False
    if _DOC_QUERY_HINT.search(q):
        return False
    return True


def safe_general_answer(query: str) -> tuple[str, str | None]:
    """Never raises. Returns (reply_text, optional debug exception summary if generation failed)."""
    try:
        return generate_general_answer(query), None
    except Exception as exc:
        return MSG_SERVICE_UNAVAILABLE, debug_service.short_exc(exc)


def _finalize_answer(turn: AssistantTurn, **dbg: Any) -> AssistantTurn:
    if debug_service.debug_enabled():
        debug_service.merge(**dbg)
    return turn


def _hits_to_excerpts(hits: list[RetrievedChunk] | None) -> list[dict[str, Any]]:
    if not hits:
        return []
    out: list[dict[str, Any]] = []
    for h in hits:
        meta = h.metadata
        page = meta.get("page_number")
        page_label = str(page) if page is not None else "-"
        out.append(
            {
                "file_name": str(meta.get("source_name") or "Unknown"),
                "chunk_id": str(meta.get("chunk_id") or ""),
                "page_label": page_label,
                "preview_text": preview_text(h.page_content),
            }
        )
    return out


def assistant_payload(res: GroundedAnswer, hits: list[RetrievedChunk] | None) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": res.answer,
        "grounded": True,
        "sources": [asdict(s) for s in res.sources],
        "excerpts": _hits_to_excerpts(hits),
    }


def _with_status_note(msg: dict[str, Any], note: str | None) -> dict[str, Any]:
    if note and note.strip():
        return {**msg, "status_note": note.strip()}
    return msg


def append_assistant_turn(
    messages: list[dict[str, Any]],
    turn: AssistantTurn,
    *,
    ingest_note: str | None = None,
) -> None:
    note = merge_notes(ingest_note, turn.assistant_note)
    if turn.mode == "error":
        messages.append(
            _with_status_note({"role": "assistant", "content": turn.error or turn.text}, note)
        )
    elif turn.mode == "general":
        messages.append(_with_status_note({"role": "assistant", "content": turn.text}, note))
    elif turn.mode == "grounded" and turn.grounded:
        messages.append(_with_status_note(assistant_payload(turn.grounded, turn.hits), note))


def _answer_user_query_impl(
    query: str,
    *,
    raw_dir: Path,
    faiss_folder: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> AssistantTurn:
    if not index_service.library_fingerprint(raw_dir):
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_no_library",
            library_fingerprint="(empty)",
            index_sync="none_no_files",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=False,
            exception_summary=gerr,
        )

    ok, _sync_msg = index_service.ensure_index_matches_library(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not ok:
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=MSG_LIBRARY_UNAVAILABLE),
            routing="general_sync_fallback",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr,
        )

    if wants_no_retrieval_fastpath(query):
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_fastpath",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=False,
            no_retrieval_fastpath=True,
            exception_summary=gerr,
        )

    try:
        store = index_service.load_faiss_store(faiss_folder)
        nvec = faiss_vector_count(store)
    except Exception as exc:
        if debug_service.debug_enabled():
            debug_service.merge(
                retrieval_load_exception=debug_service.short_exc(exc),
                retrieval_fallback_to_general=True,
            )
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_retrieval_failed",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )

    if nvec == 0:
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_empty_index",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=False,
            exception_summary=gerr,
        )

    k = min(int(top_k), nvec)
    try:
        hits = retrieve_top_k(store, query, k=k)
    except Exception as exc:
        if debug_service.debug_enabled():
            debug_service.merge(
                retrieval_query_exception=debug_service.short_exc(exc),
                retrieval_fallback_to_general=True,
            )
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_retrieval_failed",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )

    if retrieval_is_useful(hits):
        try:
            ga = generate_grounded_answer(query, hits)
        except Exception as exc:
            if debug_service.debug_enabled():
                debug_service.merge(
                    grounded_generation_exception=debug_service.short_exc(exc),
                    grounded_fallback_to_general=True,
                )
            text, gerr = safe_general_answer(query)
            exc_sum = f"{debug_service.short_exc(exc)} | {gerr}" if gerr else debug_service.short_exc(exc)
            return _finalize_answer(
                AssistantTurn(mode="general", text=text, assistant_note=MSG_GROUNDED_FALLBACK_NOTE),
                routing="general_grounded_fallback",
                retrieval_ran=True,
                retrieval_hit_count=len(hits),
                fallback_to_general=True,
                exception_summary=exc_sum,
            )
        return _finalize_answer(
            AssistantTurn(mode="grounded", text=ga.answer, grounded=ga, hits=hits),
            routing="grounded",
            retrieval_ran=True,
            retrieval_hit_count=len(hits),
            fallback_to_general=False,
        )

    text, gerr = safe_general_answer(query)
    return _finalize_answer(
        AssistantTurn(mode="general", text=text, assistant_note=None),
        routing="general_weak_retrieval",
        retrieval_ran=True,
        retrieval_hit_count=len(hits),
        fallback_to_general=True,
        exception_summary=gerr,
    )


def answer_user_query(
    query: str,
    *,
    raw_dir: Path,
    faiss_folder: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    task_mode: str = "auto",
    summarize_scope: str = "all",
) -> AssistantTurn:
    """
    Route to a general assistant reply when there is no library or retrieval is weak;
    otherwise return a document-grounded answer with sources.

    ``task_mode`` ``summarize`` / ``extract`` / ``compare`` runs document-centric prompts
    (rule-selected in the UI). ``auto`` preserves the original chat + Q&A routing.
    """
    if not query.strip():
        return _finalize_answer(
            AssistantTurn(mode="error", text=MSG_EMPTY_MESSAGE, error=MSG_EMPTY_MESSAGE),
            routing="error_empty",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=False,
        )
    try:
        if task_mode in ("summarize", "extract", "compare"):
            from app.services.doc_task_service import DocTask, run_document_task

            return run_document_task(
                query,
                cast(DocTask, task_mode),
                raw_dir=raw_dir,
                faiss_folder=faiss_folder,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                summarize_scope=summarize_scope,
            )
        return _answer_user_query_impl(
            query,
            raw_dir=raw_dir,
            faiss_folder=faiss_folder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )
    except Exception as exc:
        if debug_service.debug_enabled():
            debug_service.merge(unhandled_answer_exception=debug_service.short_exc(exc))
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_unhandled_fallback",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )
