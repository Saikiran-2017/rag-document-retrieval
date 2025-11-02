"""
Document-centric tasks: summarize, extract, compare.

Rule-based routing only (no agents). Uses the same FAISS retrieval stack as chat Q&A;
tasks request wider k and optional source filtering for single-file summarize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from app.llm.generator import (
    generate_document_task_answer,
    generate_general_answer,
    retrieval_is_useful,
)
from app.retrieval.vector_store import RetrievedChunk, faiss_vector_count, retrieve_top_k
from app.services import debug_service, index_service
from app.services.message_service import (
    MSG_GROUNDED_FALLBACK_NOTE,
    MSG_LIBRARY_UNAVAILABLE,
    merge_notes,
)

from .chat_service import AssistantTurn, _finalize_answer, safe_general_answer

DocTask = Literal["summarize", "extract", "compare"]

# Slightly relaxed gate for extract/compare when user phrasing is broad.
_RELAXED_RETRIEVAL_L2 = 1.38


def _source_name(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("source_name") or "").strip()


def _filter_by_file(hits: list[RetrievedChunk], filename: str) -> list[RetrievedChunk]:
    fn = filename.strip().lower()
    if not fn:
        return hits
    return [h for h in hits if _source_name(h).lower() == fn]


def _dedupe_preserve_order(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for h in chunks:
        cid = str(h.metadata.get("chunk_id") or h.page_content[:40])
        if cid in seen:
            continue
        seen.add(cid)
        out.append(h)
    return out


def _retrieve_broad(
    store: Any,
    query: str,
    *,
    k: int,
) -> list[RetrievedChunk]:
    nvec = faiss_vector_count(store)
    if nvec == 0:
        return []
    kk = max(1, min(int(k), nvec))
    return retrieve_top_k(store, query, k=kk)


def _hits_for_summarize(
    store: Any,
    user_query: str,
    *,
    top_k: int,
    summarize_scope: str,
) -> tuple[list[RetrievedChunk], str | None]:
    """
    Retrieve chunks for summarization. If summarize_scope is a filename, bias toward that file.
    Returns (hits, optional assistant_note).
    """
    nvec = faiss_vector_count(store)
    k_target = min(nvec, max(10, int(top_k) * 2))
    base_q = (user_query or "").strip() or "main themes key points sections overview"

    hits = _retrieve_broad(store, base_q, k=k_target)
    note: str | None = None

    scope = (summarize_scope or "all").strip().lower()
    if scope and scope != "all":
        filtered = _filter_by_file(hits, summarize_scope)
        if len(filtered) < max(3, min(5, k_target // 2)):
            hits2 = _retrieve_broad(store, f"content from {summarize_scope}", k=min(nvec, k_target * 2))
            merged = _dedupe_preserve_order(filtered + hits2)
            filtered = _filter_by_file(merged, summarize_scope)[:k_target]
        if not filtered:
            return [], "No passages from that file were found after sync. Try another file or summarize all documents."
        hits = filtered[:k_target]
        if hits and float(hits[0].distance) > 1.45:
            note = "Summary uses retrieved excerpts from that file; ask a follow-up to go deeper."

    if not scope or scope == "all":
        if hits and float(hits[0].distance) > 1.45:
            note = "Summary is based on retrieved excerpts across your library; coverage may be partial."

    return hits, note


def _hits_for_extract(
    store: Any,
    user_query: str,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    nvec = faiss_vector_count(store)
    k_target = min(nvec, max(6, int(top_k)))
    q = user_query.strip() or "deadlines action items requirements decisions entities"
    return _retrieve_broad(store, q, k=k_target)


def _hits_for_compare(
    store: Any,
    user_query: str,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    nvec = faiss_vector_count(store)
    k_target = min(nvec, max(10, int(top_k) * 2))
    q = user_query.strip() or "differences similarities changes across documents"
    return _retrieve_broad(store, q, k=k_target)


def _distinct_sources(hits: list[RetrievedChunk]) -> set[str]:
    return {s for s in (_source_name(h) for h in hits) if s}


def run_document_task(
    query: str,
    task: DocTask,
    *,
    raw_dir: Path,
    faiss_folder: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    summarize_scope: str = "all",
) -> AssistantTurn:
    """
    Run summarize, extract, or compare over synced documents.

    Preconditions and retrieval quality are handled explicitly; failures fall back to
    a calm general reply when appropriate (same resilience pattern as chat_service).
    """
    if not index_service.library_fingerprint(raw_dir):
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(
                mode="general",
                text=text,
                assistant_note="Add and sync documents to use Summarize, Extract, or Compare.",
            ),
            routing=f"task_{task}_no_library",
            document_task=task,
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
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
            routing=f"task_{task}_sync_fallback",
            document_task=task,
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr,
        )

    if task == "compare":
        files = index_service.list_raw_files(raw_dir)
        if len(files) < 2:
            text, gerr = safe_general_answer(query)
            return _finalize_answer(
                AssistantTurn(
                    mode="general",
                    text=text,
                    assistant_note="Compare needs at least two documents in your library. Upload another file, sync, and try again.",
                ),
                routing="task_compare_need_two",
                document_task=task,
                retrieval_ran=False,
                retrieval_hit_count=0,
                fallback_to_general=True,
                exception_summary=gerr,
            )

    try:
        store = index_service.load_faiss_store(faiss_folder)
        nvec = faiss_vector_count(store)
    except Exception as exc:
        if debug_service.debug_enabled():
            debug_service.merge(
                document_task=task,
                retrieval_load_exception=debug_service.short_exc(exc),
            )
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing=f"task_{task}_store_failed",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )

    if nvec == 0:
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing=f"task_{task}_empty_index",
            document_task=task,
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=False,
            exception_summary=gerr,
        )

    assistant_note: str | None = None
    try:
        if task == "summarize":
            hits, note = _hits_for_summarize(store, query, top_k=top_k, summarize_scope=summarize_scope)
            assistant_note = note
        elif task == "extract":
            hits = _hits_for_extract(store, query, top_k=top_k)
        else:
            hits = _hits_for_compare(store, query, top_k=top_k)
    except Exception as exc:
        if debug_service.debug_enabled():
            debug_service.merge(
                document_task=task,
                retrieval_query_exception=debug_service.short_exc(exc),
            )
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing=f"task_{task}_retrieve_failed",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )

    if task == "summarize" and not hits:
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(
                mode="general",
                text=text,
                assistant_note=assistant_note or "Could not load excerpts for that summary.",
            ),
            routing="task_summarize_no_hits",
            document_task=task,
            retrieval_ran=True,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr,
        )

    useful = retrieval_is_useful(hits)
    if not useful and task in ("extract", "compare"):
        useful = retrieval_is_useful(hits, max_l2_distance=_RELAXED_RETRIEVAL_L2)
    if not useful and task == "extract":
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(
                mode="general",
                text=text,
                assistant_note="No strong matches in your documents for that request. Rephrase or sync updated files.",
            ),
            routing="task_extract_weak",
            document_task=task,
            retrieval_ran=True,
            retrieval_hit_count=len(hits),
            fallback_to_general=True,
            exception_summary=gerr,
        )
    if not useful and task == "compare":
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(
                mode="general",
                text=text,
                assistant_note="Retrieval did not surface strong passages to compare. Try a more specific question.",
            ),
            routing="task_compare_weak",
            document_task=task,
            retrieval_ran=True,
            retrieval_hit_count=len(hits),
            fallback_to_general=True,
            exception_summary=gerr,
        )

    if task == "compare" and len(_distinct_sources(hits)) < 2:
        assistant_note = merge_notes(
            assistant_note,
            "Retrieved passages mostly come from one document; comparison may be limited.",
        )

    try:
        ga = generate_document_task_answer(task, query, hits)
    except Exception as exc:
        if debug_service.debug_enabled():
            debug_service.merge(
                document_task=task,
                doc_task_generation_exception=debug_service.short_exc(exc),
            )
        text, gerr = safe_general_answer(query)
        exc_sum = f"{debug_service.short_exc(exc)} | {gerr}" if gerr else debug_service.short_exc(exc)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=MSG_GROUNDED_FALLBACK_NOTE),
            routing=f"task_{task}_gen_failed",
            document_task=task,
            retrieval_ran=True,
            retrieval_hit_count=len(hits),
            fallback_to_general=True,
            exception_summary=exc_sum,
        )

    return _finalize_answer(
        AssistantTurn(mode="grounded", text=ga.answer, grounded=ga, hits=hits, assistant_note=assistant_note),
        routing=f"task_{task}",
        document_task=task,
        retrieval_ran=True,
        retrieval_hit_count=len(hits),
        fallback_to_general=False,
    )
