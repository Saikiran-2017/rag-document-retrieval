"""
Document-centric tasks: summarize, extract, compare.

Uses the same hybrid (BM25 + vector + RRF) + rerank + context selection pipeline as chat Q&A.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from app.llm.generator import generate_document_task_answer
from app.retrieval.context_selection import (
    hybrid_pool_size,
    rerank_hybrid_hits,
    select_generation_context,
)
from app.retrieval.hybrid_retrieve import hybrid_retrieve
from app.retrieval.vector_store import RetrievedChunk, faiss_vector_count
from app.services import debug_service, document_health, index_service
from app.services.message_service import (
    MSG_GROUNDED_FALLBACK_NOTE,
    MSG_LIBRARY_UNAVAILABLE,
    merge_notes,
)

from .chat_service import AssistantTurn, _finalize_answer, safe_general_answer

DocTask = Literal["summarize", "extract", "compare"]


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


def _hybrid_ranked(
    store: Any,
    query: str,
    *,
    top_k: int,
    faiss_folder: Path,
) -> list[RetrievedChunk]:
    nvec = faiss_vector_count(store)
    if nvec == 0:
        return []
    q = (query or "").strip() or "document"
    kp = hybrid_pool_size(nvec, top_k)
    pool = hybrid_retrieve(
        store,
        q,
        k_final=min(nvec, kp),
        k_vector=min(28, nvec),
        k_bm25=min(28, nvec),
    )
    ranked = rerank_hybrid_hits(pool)
    return document_health.filter_trusted_retrieval_hits(faiss_folder, ranked)


def _hits_for_summarize(
    store: Any,
    user_query: str,
    *,
    top_k: int,
    summarize_scope: str,
    faiss_folder: Path,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], str | None]:
    """
    Returns (context_hits for LLM, ranked_full for debug/gating, optional note).

    ``ranked_full`` is the reranked hybrid list before file filter / context trim.
    """
    nvec = faiss_vector_count(store)
    base_q = (user_query or "").strip() or "main themes key points sections overview"
    ranked = _hybrid_ranked(store, base_q, top_k=top_k, faiss_folder=faiss_folder)
    note: str | None = None

    scope = (summarize_scope or "all").strip().lower()
    if scope and scope != "all":
        filtered = _filter_by_file(ranked, summarize_scope)
        if len(filtered) < max(3, min(5, top_k)):
            ranked2 = _hybrid_ranked(
                store, f"content sections text from {summarize_scope}", top_k=top_k, faiss_folder=faiss_folder
            )
            merged = _dedupe_preserve_order(filtered + _filter_by_file(ranked2, summarize_scope))
            filtered = merged
        if not filtered:
            return [], ranked, "No passages from that file were found after sync. Try another file or summarize all documents."
        ranked = filtered
        if ranked and float(ranked[0].distance) > 1.38:
            note = "Summary uses retrieved excerpts from that file; coverage may be partial."

    if not scope or scope == "all":
        if ranked and float(ranked[0].distance) > 1.38:
            note = "Summary is based on retrieved excerpts across your library; coverage may be partial."

    hits = select_generation_context(ranked, mode="summarize", top_k=top_k, nvec=nvec)
    return hits, ranked, note


def _hits_for_extract(
    store: Any,
    user_query: str,
    *,
    top_k: int,
    faiss_folder: Path,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    nvec = faiss_vector_count(store)
    q = user_query.strip() or "deadlines action items requirements decisions entities"
    ranked = _hybrid_ranked(store, q, top_k=top_k, faiss_folder=faiss_folder)
    hits = select_generation_context(ranked, mode="extract", top_k=top_k, nvec=nvec)
    return hits, ranked


def _hits_for_compare(
    store: Any,
    user_query: str,
    *,
    top_k: int,
    faiss_folder: Path,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    nvec = faiss_vector_count(store)
    q = user_query.strip() or "differences similarities changes across documents"
    ranked = _hybrid_ranked(store, q, top_k=top_k, faiss_folder=faiss_folder)
    hits = select_generation_context(ranked, mode="compare", top_k=top_k, nvec=nvec)
    return hits, ranked


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
    if not index_service.list_raw_files(raw_dir):
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
    ranked_for_gate: list[RetrievedChunk] = []
    try:
        if task == "summarize":
            hits, ranked_for_gate, note = _hits_for_summarize(
                store,
                query,
                top_k=top_k,
                summarize_scope=summarize_scope,
                faiss_folder=faiss_folder,
            )
            assistant_note = note
        elif task == "extract":
            hits, ranked_for_gate = _hits_for_extract(
                store, query, top_k=top_k, faiss_folder=faiss_folder
            )
        else:
            hits, ranked_for_gate = _hits_for_compare(
                store, query, top_k=top_k, faiss_folder=faiss_folder
            )
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

    if not hits and task in ("extract", "compare"):
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(
                mode="general",
                text=text,
                assistant_note="No document excerpts were selected for this task. Try Sync or a different question.",
            ),
            routing=f"task_{task}_no_context",
            document_task=task,
            retrieval_ran=True,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr,
        )

    useful = document_health.allow_document_grounding(
        faiss_folder, ranked_for_gate, for_document_task=True
    )
    if not useful:
        text, gerr = safe_general_answer(query)
        msg = {
            "summarize": "Retrieval did not find strong matches for a reliable summary. Rephrase, add files, or try a narrower question.",
            "extract": "No strong matches in your documents for that request. Rephrase or sync updated files.",
            "compare": "Retrieval did not surface strong passages to compare. Try a more specific question.",
        }[task]
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=msg),
            routing=f"task_{task}_weak_hybrid",
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

    if debug_service.debug_enabled():
        debug_service.merge(
            document_task=task,
            task_context_chunk_count=len(hits),
            task_ranked_pool_size=len(ranked_for_gate),
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
        AssistantTurn(
            mode="grounded",
            text=ga.answer,
            grounded=ga,
            hits=hits,
            assistant_note=merge_notes(assistant_note, ga.validation_warning),
        ),
        routing=f"task_{task}",
        document_task=task,
        retrieval_ran=True,
        retrieval_hit_count=len(hits),
        fallback_to_general=False,
    )
