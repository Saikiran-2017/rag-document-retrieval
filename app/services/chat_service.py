"""Query routing, grounded vs general answers, safe fallbacks, and assistant message shaping."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from app.llm.answer_validation import validate_grounded_answer, validate_web_markdown_links
from app.llm.generator import (
    UNKNOWN_PHRASE,
    GroundedAnswer,
    chunks_to_source_refs,
    generate_blended_answer,
    generate_general_answer,
    generate_grounded_answer,
    generate_web_grounded_answer,
    stream_blended_answer_tokens,
    stream_general_answer_tokens,
    stream_grounded_answer_tokens,
    stream_web_grounded_answer_tokens,
)
from app.llm.query_rewrite import rewrite_for_retrieval
from app.retrieval.context_selection import (
    hybrid_pool_size,
    rerank_hybrid_hits,
    select_generation_context,
)
from app.retrieval.hybrid_retrieve import hybrid_retrieve
from app.retrieval.vector_store import RetrievedChunk, faiss_vector_count
from app.services import debug_service, document_health, index_service
from app.services.perf_service import record_phase_ms, timed_phase
from app.services.web_search_service import prepare_web_for_generation, web_results_strong_enough
from app.services.message_service import (
    MSG_EMPTY_MESSAGE,
    MSG_GROUNDED_FALLBACK_NOTE,
    MSG_LIBRARY_UNAVAILABLE,
    MSG_SERVICE_UNAVAILABLE,
    MSG_WEB_RESULTS_THIN,
    merge_notes,
    preview_text,
)


@dataclass
class AssistantTurn:
    """Result of routing: document-grounded, web, blended, or general."""

    mode: Literal["grounded", "general", "error", "web", "blended"]
    text: str
    grounded: GroundedAnswer | None = None
    hits: list[RetrievedChunk] | None = None
    error: str | None = None
    # Muted status line under the answer (library/sync/grounded-fallback hints).
    assistant_note: str | None = None
    # Web snippets used for web-only or blended answers (title, url, snippet).
    web_snippets: list[dict[str, str]] | None = None
    # When set, UI should ``st.write_stream(stream_tokens())`` then :func:`materialize_streamed_turn`.
    stream_tokens: Callable[[], Iterator[str]] | None = None


def _streaming_enabled() -> bool:
    return os.environ.get("KA_NO_STREAM", "").strip().lower() not in ("1", "true", "yes")


def materialize_streamed_turn(turn: AssistantTurn, full_text: str) -> AssistantTurn:
    """After streaming completes, fill ``text`` / ``grounded`` for persistence and expanders."""
    if not turn.stream_tokens:
        return turn
    ft = (full_text or "").strip()
    if turn.mode == "grounded" and turn.hits:
        raw = ft or UNKNOWN_PHRASE
        fixed, warn = validate_grounded_answer(raw, turn.hits, unknown_phrase=UNKNOWN_PHRASE)
        ga = GroundedAnswer(
            answer=fixed,
            sources=chunks_to_source_refs(turn.hits),
            validation_warning=warn,
        )
        return AssistantTurn(
            mode="grounded",
            text=ga.answer,
            grounded=ga,
            hits=turn.hits,
            assistant_note=merge_notes(turn.assistant_note, warn),
            web_snippets=turn.web_snippets,
            stream_tokens=None,
        )
    if turn.mode == "blended" and turn.hits:
        raw = ft or UNKNOWN_PHRASE
        fixed, warn = validate_grounded_answer(raw, turn.hits, unknown_phrase=UNKNOWN_PHRASE)
        wurls = [str(d.get("url") or "") for d in (turn.web_snippets or [])]
        wfixed, wwarn = validate_web_markdown_links(fixed, wurls)
        ga = GroundedAnswer(
            answer=wfixed,
            sources=chunks_to_source_refs(turn.hits),
            validation_warning=merge_notes(warn, wwarn),
        )
        return AssistantTurn(
            mode="blended",
            text=ga.answer,
            grounded=ga,
            hits=turn.hits,
            web_snippets=turn.web_snippets,
            assistant_note=merge_notes(turn.assistant_note, ga.validation_warning),
            stream_tokens=None,
        )
    if turn.mode == "web":
        wurls = [str(d.get("url") or "") for d in (turn.web_snippets or [])]
        wfixed, wwarn = validate_web_markdown_links(ft or "", wurls)
        return AssistantTurn(
            mode="web",
            text=wfixed or "No answer generated.",
            web_snippets=turn.web_snippets,
            assistant_note=merge_notes(turn.assistant_note, wwarn),
            stream_tokens=None,
        )
    if turn.mode == "general":
        return AssistantTurn(
            mode="general",
            text=ft or "I don't have a response right now.",
            assistant_note=turn.assistant_note,
            stream_tokens=None,
        )
    return AssistantTurn(
        mode=turn.mode,
        text=ft or turn.text,
        grounded=turn.grounded,
        hits=turn.hits,
        error=turn.error,
        assistant_note=turn.assistant_note,
        web_snippets=turn.web_snippets,
        stream_tokens=None,
    )


# If the query clearly does not ask about documents, skip FAISS load + retrieval (same UX).
# When documents are strong, still pull web snippets for time-sensitive or external facts.
_TIME_SENSITIVE_HINT = re.compile(
    r"\b(today|latest|news|current|recent|price|stock|market|weather|breaking|"
    r"who is|who was|when did|2024|2025|2026)\b",
    re.I,
)


def _wants_blended_web(query: str) -> bool:
    return bool(_TIME_SENSITIVE_HINT.search(query))


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
    from app.reliability.turn_log import log_reliability_turn

    log_reliability_turn(turn, **dbg)
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
    elif turn.mode == "web":
        pl: dict[str, Any] = {"role": "assistant", "content": turn.text, "web_only": True}
        if turn.web_snippets:
            pl["web_sources"] = turn.web_snippets
        messages.append(_with_status_note(pl, note))
    elif turn.mode == "blended" and turn.grounded:
        pl = assistant_payload(turn.grounded, turn.hits)
        pl["blended"] = True
        if turn.web_snippets:
            pl["web_sources"] = turn.web_snippets
        messages.append(_with_status_note(pl, note))
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
    if not index_service.list_raw_files(raw_dir):
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
        if _streaming_enabled():
            return _finalize_answer(
                AssistantTurn(
                    mode="general",
                    text="",
                    assistant_note=None,
                    stream_tokens=lambda: stream_general_answer_tokens(query),
                ),
                routing="general_fastpath",
                retrieval_ran=False,
                retrieval_hit_count=0,
                fallback_to_general=False,
                no_retrieval_fastpath=True,
                exception_summary=None,
            )
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

    k_pool = hybrid_pool_size(nvec, int(top_k))
    try:
        t_rw = time.perf_counter()
        rewritten = rewrite_for_retrieval(query)
        record_phase_ms("rewrite", (time.perf_counter() - t_rw) * 1000)
        with timed_phase("retrieval"):
            pool_hits = hybrid_retrieve(
                store,
                rewritten,
                k_final=k_pool,
                k_vector=min(28, nvec),
                k_bm25=min(28, nvec),
            )
            ranked_hits = rerank_hybrid_hits(pool_hits)
            ranked_hits = document_health.filter_trusted_retrieval_hits(faiss_folder, ranked_hits)
            doc_good = document_health.allow_document_grounding(faiss_folder, ranked_hits)
            hits = (
                select_generation_context(ranked_hits, mode="qa", top_k=int(top_k), nvec=nvec)
                if doc_good
                else []
            )
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

    if debug_service.debug_enabled():
        top = hits[0] if hits else (ranked_hits[0] if ranked_hits else None)
        debug_service.merge(
            rewritten_query=rewritten[:220],
            retrieval_pool_size=len(pool_hits),
            retrieval_context_size=len(hits),
            hybrid_hit_count=len(hits),
            retrieval_rerank_top_score=float(top.metadata.get("rerank_score", 0)) if top else None,
            best_distance=float(top.distance) if top else None,
            best_rrf=float(top.metadata.get("rrf_score", 0)) if top else None,
        )

    web_snippets_list, web_block, web_dicts, shaped_web_q = prepare_web_for_generation(query, rewritten)
    web_strong = web_results_strong_enough(web_snippets_list, shaped_query=shaped_web_q)
    web_allowed_urls = [s.url for s in web_snippets_list]
    stream = _streaming_enabled()

    if debug_service.debug_enabled():
        debug_service.merge(
            web_snippet_count=len(web_snippets_list),
            web_results_strong=web_strong,
        )

    def _gen_web(note: str | None = None) -> AssistantTurn:
        if stream:

            def _tok() -> Iterator[str]:
                yield from stream_web_grounded_answer_tokens(query, web_block)

            return AssistantTurn(
                mode="web",
                text="",
                web_snippets=web_dicts,
                assistant_note=note,
                stream_tokens=_tok,
            )
        with timed_phase("generation"):
            ans = generate_web_grounded_answer(query, web_block)
        fixed, wwarn = validate_web_markdown_links(ans, web_allowed_urls)
        return AssistantTurn(
            mode="web",
            text=fixed,
            web_snippets=web_dicts,
            assistant_note=merge_notes(note, wwarn),
        )

    def _gen_blend() -> AssistantTurn:
        blend_note = (
            "Documents: facts from your library are cited as [SOURCE n]. "
            "Web: time-sensitive or external facts use the linked URLs shown under Web sources."
        )
        if stream:

            def _tok() -> Iterator[str]:
                yield from stream_blended_answer_tokens(query, hits, web_block)

            return AssistantTurn(
                mode="blended",
                text="",
                grounded=None,
                hits=hits,
                web_snippets=web_dicts,
                assistant_note=blend_note,
                stream_tokens=_tok,
            )
        with timed_phase("generation"):
            ga = generate_blended_answer(query, hits, web_block)
        return AssistantTurn(
            mode="blended",
            text=ga.answer,
            grounded=ga,
            hits=hits,
            web_snippets=web_dicts,
            assistant_note=merge_notes(blend_note, ga.validation_warning),
        )

    try:
        if doc_good and web_snippets_list and web_strong and _wants_blended_web(query):
            turn = _gen_blend()
            return _finalize_answer(
                turn,
                routing="blended",
                retrieval_ran=True,
                retrieval_hit_count=len(hits),
                fallback_to_general=False,
            )
        if doc_good:
            if stream:

                def _gtok() -> Iterator[str]:
                    yield from stream_grounded_answer_tokens(query, hits)

                return _finalize_answer(
                    AssistantTurn(
                        mode="grounded",
                        text="",
                        grounded=None,
                        hits=hits,
                        stream_tokens=_gtok,
                    ),
                    routing="grounded",
                    retrieval_ran=True,
                    retrieval_hit_count=len(hits),
                    fallback_to_general=False,
                )
            try:
                with timed_phase("generation"):
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
                AssistantTurn(
                    mode="grounded",
                    text=ga.answer,
                    grounded=ga,
                    hits=hits,
                    assistant_note=ga.validation_warning,
                ),
                routing="grounded",
                retrieval_ran=True,
                retrieval_hit_count=len(hits),
                fallback_to_general=False,
            )
        if web_snippets_list and web_strong:
            return _finalize_answer(
                _gen_web("No strong match in your documents; answer uses web results."),
                routing="web_weak_docs",
                retrieval_ran=True,
                retrieval_hit_count=len(hits),
                fallback_to_general=True,
            )
        if web_snippets_list:
            text, gerr = safe_general_answer(query)
            return _finalize_answer(
                AssistantTurn(
                    mode="general",
                    text=text,
                    assistant_note=MSG_WEB_RESULTS_THIN,
                ),
                routing="general_web_thin",
                retrieval_ran=True,
                retrieval_hit_count=len(hits),
                fallback_to_general=True,
                exception_summary=gerr,
            )
    except Exception as exc:
        if debug_service.debug_enabled():
            debug_service.merge(generation_route_exception=debug_service.short_exc(exc))
        text, gerr = safe_general_answer(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=MSG_GROUNDED_FALLBACK_NOTE),
            routing="route_generation_failed",
            retrieval_ran=True,
            retrieval_hit_count=len(hits),
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )

    if stream:
        return _finalize_answer(
            AssistantTurn(
                mode="general",
                text="",
                assistant_note="No strong document match and no web results; general answer only.",
                stream_tokens=lambda: stream_general_answer_tokens(query),
            ),
            routing="general_weak_no_web",
            retrieval_ran=True,
            retrieval_hit_count=len(hits),
            fallback_to_general=True,
            exception_summary=None,
        )
    text, gerr = safe_general_answer(query)
    return _finalize_answer(
        AssistantTurn(
            mode="general",
            text=text,
            assistant_note="No strong document match and no web results; general answer only.",
        ),
        routing="general_weak_no_web",
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
