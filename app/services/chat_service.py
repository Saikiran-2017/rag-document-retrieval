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
    generate_document_abstain_answer,
    generate_general_answer,
    generate_grounded_answer,
    generate_web_grounded_answer,
    stream_blended_answer_tokens,
    stream_document_abstain_tokens,
    stream_general_answer_tokens,
    stream_grounded_answer_tokens,
    stream_web_grounded_answer_tokens,
)
from app.llm.query_intent import (
    is_broad_document_overview_query,
    is_section_navigation_query,
    is_sparse_entity_lookup_query,
    user_expects_document_grounding,
    uses_relaxed_document_grounding_gate,
)
from app.llm.query_rewrite import rewrite_for_retrieval
from app.retrieval.context_selection import (
    hybrid_pool_size,
    prioritize_section_navigation_hits,
    rerank_hybrid_hits,
    select_generation_context,
)
from app.retrieval.hybrid_retrieve import hybrid_retrieve, merge_hybrid_hit_pools
from app.retrieval.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    RetrievedChunk,
    faiss_index_files_exist,
    faiss_vector_count,
)
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
    # Optional developer diagnostics (only populated in debug mode).
    diagnostics: dict[str, Any] | None = None


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
            diagnostics=turn.diagnostics,
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
            diagnostics=turn.diagnostics,
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
    # "Who is the CFO" matches time-sensitive heuristics but should stay document-grounded when
    # library hits exist (avoid unrelated web CFO news in blended answers).
    if is_sparse_entity_lookup_query(query):
        return False
    return bool(_TIME_SENSITIVE_HINT.search(query))


# Second-pass retrieval query for broad doc questions (improves recall across sections).
_DOC_OVERVIEW_RETRIEVAL_BOOST = (
    "main sections themes introduction conclusion purpose overview key ideas summary"
)


def wants_no_retrieval_fastpath(query: str) -> bool:
    q = query.strip()
    if len(q) < 4 or len(q) > 220:
        return False
    if user_expects_document_grounding(q):
        return False
    return True


def safe_general_answer(query: str) -> tuple[str, str | None]:
    """Never raises. Returns (reply_text, optional debug exception summary if generation failed)."""
    try:
        return generate_general_answer(query), None
    except Exception as exc:
        return MSG_SERVICE_UNAVAILABLE, debug_service.short_exc(exc)


def safe_document_abstain_answer(query: str) -> tuple[str, str | None]:
    """When the user expected library-backed answers but retrieval is weak; never raises."""
    try:
        return generate_document_abstain_answer(query), None
    except Exception as exc:
        return UNKNOWN_PHRASE, debug_service.short_exc(exc)


def _general_or_abstain(query: str) -> tuple[str, str | None]:
    """General chat only when the question is not clearly document-scoped."""
    if user_expects_document_grounding(query):
        return safe_document_abstain_answer(query)
    return safe_general_answer(query)


def _hit_debug_rows(hits: list[RetrievedChunk], limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, h in enumerate(hits[:limit]):
        rows.append(
            {
                "rank": i,
                "source": str(h.metadata.get("source_name") or ""),
                "l2": round(float(h.distance), 4),
                "rrf": round(float(h.metadata.get("rrf_score", 0) or 0), 5),
                "preview": ((h.page_content or "")[:120]).replace("\n", " "),
            }
        )
    return rows


def _finalize_answer(turn: AssistantTurn, **dbg: Any) -> AssistantTurn:
    if debug_service.debug_enabled():
        debug_service.merge(**dbg)
        # Developer-only diagnostics payload for API/UI.
        srcs = []
        if turn.hits:
            seen: set[str] = set()
            for h in turn.hits:
                sn = str(h.metadata.get("source_name") or "").strip()
                if sn and sn not in seen:
                    seen.add(sn)
                    srcs.append(sn)
        diag: dict[str, Any] = {
            "mode": turn.mode,
            "routing": dbg.get("routing"),
            "route_selected": dbg.get("route_selected") or dbg.get("routing"),
            "pool_size": dbg.get("pool_size"),
            "after_rerank": dbg.get("after_rerank"),
            "after_trust_filter": dbg.get("after_trust_filter"),
            "context_chunks_selected": dbg.get("context_chunks_selected"),
            "grounding_gate_reason": dbg.get("grounding_gate_reason"),
            "selected_sources": srcs,
            "retrieval_hit_count": dbg.get("retrieval_hit_count"),
            "retrieval_ran": dbg.get("retrieval_ran"),
            "fallback_to_general": dbg.get("fallback_to_general"),
            "exception_summary": dbg.get("exception_summary"),
        }
        # Keep raw debug event rows when present (bounded in size already).
        for k in ("top_pool_hits", "top_ranked_pre_trust", "top_trusted_hits"):
            if k in dbg:
                diag[k] = dbg.get(k)
        # Attach to turn and store as last snapshot for debug endpoints.
        turn = AssistantTurn(**{**turn.__dict__, "diagnostics": diag})
        debug_service.set_last_diagnostics(diag)
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
    debug_service.log_retrieval_event(
        "turn_begin",
        raw_dir=str(raw_dir.resolve()),
        faiss_folder=str(faiss_folder.resolve()),
        index_on_disk=faiss_index_files_exist(faiss_folder, index_name=DEFAULT_INDEX_NAME),
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        top_k=int(top_k),
    )

    if not index_service.list_raw_files(raw_dir):
        text, gerr = _general_or_abstain(query)
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
        debug_service.log_retrieval_event("index_sync_failed", sync_message=_sync_msg[:300])
        text, gerr = _general_or_abstain(query)
        return _finalize_answer(
            AssistantTurn(
                mode="general",
                text=text,
                assistant_note=(_sync_msg or MSG_LIBRARY_UNAVAILABLE),
            ),
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
        text, gerr = _general_or_abstain(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_retrieval_failed",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )

    if nvec == 0:
        debug_service.log_retrieval_event("empty_vector_index", faiss_folder=str(faiss_folder.resolve()))
        text, gerr = _general_or_abstain(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_empty_index",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=False,
            exception_summary=gerr,
        )

    debug_service.log_retrieval_event(
        "index_loaded",
        vector_count=int(nvec),
        faiss_folder=str(faiss_folder.resolve()),
    )

    # Diagnostics captured for debug mode (and optionally attached to responses).
    diag_pool_size: int | None = None
    diag_after_rerank: int | None = None
    diag_after_trust: int | None = None
    diag_ctx_selected: int | None = None
    diag_gate_reason: str | None = None

    base_pool = hybrid_pool_size(nvec, int(top_k))
    broad_overview = is_broad_document_overview_query(query)
    relaxed_doc_gate = uses_relaxed_document_grounding_gate(query)
    section_nav = is_section_navigation_query(query)
    lookup_relaxed = is_sparse_entity_lookup_query(query)
    wide_retrieval = broad_overview or relaxed_doc_gate or section_nav
    if wide_retrieval:
        k_pool = min(nvec, max(base_pool, 22))
        kv = min(nvec, max(32, min(48, nvec)))
    else:
        k_pool = base_pool
        kv = min(28, nvec)
    try:
        t_rw = time.perf_counter()
        rewritten = rewrite_for_retrieval(query)
        record_phase_ms("rewrite", (time.perf_counter() - t_rw) * 1000)
        with timed_phase("retrieval"):
            pool_hits = hybrid_retrieve(
                store,
                rewritten,
                k_final=k_pool,
                k_vector=kv,
                k_bm25=kv,
            )
            if wide_retrieval:
                boost_q = f"{rewritten} {_DOC_OVERVIEW_RETRIEVAL_BOOST}".strip()
                if relaxed_doc_gate and not broad_overview:
                    boost_q = (
                        f"{boost_q} performance latency p99 workload reliability metrics discussion"
                    ).strip()
                if section_nav and not broad_overview:
                    boost_q = (
                        f"{boost_q} section heading chapter appendix subsection disaster recovery"
                    ).strip()
                pool_b = hybrid_retrieve(
                    store,
                    boost_q,
                    k_final=k_pool,
                    k_vector=kv,
                    k_bm25=kv,
                )
                pool_hits = merge_hybrid_hit_pools(pool_hits, pool_b)
            ranked_hits = rerank_hybrid_hits(pool_hits)
            if section_nav:
                ranked_hits = prioritize_section_navigation_hits(ranked_hits, query)
            n_after_rerank = len(ranked_hits)
            trusted_hits = document_health.filter_trusted_retrieval_hits(faiss_folder, ranked_hits)
            doc_good, gate_reason = document_health.explain_allow_document_grounding(
                faiss_folder,
                trusted_hits,
                relaxed_doc_qa=relaxed_doc_gate,
                lookup_qa_relaxed=lookup_relaxed,
            )
            hits = (
                select_generation_context(
                    trusted_hits,
                    mode="qa",
                    top_k=int(top_k),
                    nvec=nvec,
                    broad_document_question=broad_overview or relaxed_doc_gate or section_nav,
                    section_navigation_query=section_nav,
                    focus_source_name=str(trusted_hits[0].metadata.get("source_name") or "").strip()
                    if (trusted_hits and (lookup_relaxed or section_nav))
                    else None,
                )
                if doc_good
                else []
            )
            debug_service.log_retrieval_event(
                "retrieval_hybrid_done",
                user_query_preview=query[:200],
                rewritten_query=rewritten[:300],
                broad_overview=broad_overview,
                wide_retrieval=wide_retrieval,
                relaxed_doc_gate=relaxed_doc_gate,
                section_navigation=section_nav,
                lookup_qa_relaxed=lookup_relaxed,
                pool_size=len(pool_hits),
                after_rerank=n_after_rerank,
                after_trust_filter=len(trusted_hits),
                context_chunks_selected=len(hits),
                doc_grounding_allowed=doc_good,
                grounding_gate_reason=gate_reason,
                top_pool_hits=_hit_debug_rows(pool_hits, 8),
                top_ranked_pre_trust=_hit_debug_rows(ranked_hits, 8),
                top_trusted_hits=_hit_debug_rows(trusted_hits, 8),
            )
            diag_pool_size = len(pool_hits)
            diag_after_rerank = n_after_rerank
            diag_after_trust = len(trusted_hits)
            diag_ctx_selected = len(hits)
            diag_gate_reason = gate_reason
            ranked_hits = trusted_hits
    except Exception as exc:
        if debug_service.debug_enabled():
            debug_service.merge(
                retrieval_query_exception=debug_service.short_exc(exc),
                retrieval_fallback_to_general=True,
            )
        debug_service.log_retrieval_event(
            "retrieval_exception", error=debug_service.short_exc(exc)
        )
        text, gerr = _general_or_abstain(query)
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

    # Web search can be slow; avoid it on the common grounded-doc path unless blending is desired.
    wants_web = (not doc_good) or _wants_blended_web(query)
    if wants_web:
        web_snippets_list, web_block, web_dicts, shaped_web_q = prepare_web_for_generation(query, rewritten)
        web_strong = web_results_strong_enough(web_snippets_list, shaped_query=shaped_web_q)
        web_allowed_urls = [s.url for s in web_snippets_list]
    else:
        web_snippets_list, web_block, web_dicts, shaped_web_q = [], "", None, ""
        web_strong = False
        web_allowed_urls = []
    stream = _streaming_enabled()

    if debug_service.debug_enabled():
        debug_service.merge(
            web_snippet_count=len(web_snippets_list),
            web_results_strong=web_strong,
        )

    if doc_good and hits:
        _route = "grounded_or_blended"
    elif doc_good and not hits:
        _route = "doc_allowed_empty_context"
    elif web_snippets_list and web_strong:
        _route = "web_or_blend"
    else:
        _route = "general_weak_or_abstain"
    debug_service.log_retrieval_event(
        "routing_decision",
        route=_route,
        doc_grounding_allowed=doc_good,
        context_hits_for_llm=len(hits),
        web_strong=web_strong,
        web_snippet_count=len(web_snippets_list),
        llm_gets_grounded_prompt=bool(doc_good and hits),
    )

    common_diag = {
        "route_selected": _route,
        "pool_size": diag_pool_size,
        "after_rerank": diag_after_rerank,
        "after_trust_filter": diag_after_trust,
        "context_chunks_selected": diag_ctx_selected,
        "grounding_gate_reason": diag_gate_reason,
    }

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
                **common_diag,
            )
        if doc_good:
            if stream:

                def _gtok() -> Iterator[str]:
                    yield from stream_grounded_answer_tokens(
                        query, hits, section_navigation_query=section_nav
                    )

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
                    **common_diag,
                )
            try:
                with timed_phase("generation"):
                    ga = generate_grounded_answer(
                        query, hits, section_navigation_query=section_nav
                    )
            except Exception as exc:
                if debug_service.debug_enabled():
                    debug_service.merge(
                        grounded_generation_exception=debug_service.short_exc(exc),
                        grounded_fallback_to_general=True,
                    )
                text, gerr = (
                    safe_document_abstain_answer(query)
                    if user_expects_document_grounding(query)
                    else safe_general_answer(query)
                )
                exc_sum = f"{debug_service.short_exc(exc)} | {gerr}" if gerr else debug_service.short_exc(exc)
                return _finalize_answer(
                    AssistantTurn(mode="general", text=text, assistant_note=MSG_GROUNDED_FALLBACK_NOTE),
                    routing="general_grounded_fallback",
                    retrieval_ran=True,
                    retrieval_hit_count=len(hits),
                    fallback_to_general=True,
                    exception_summary=exc_sum,
                    **common_diag,
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
                **common_diag,
            )
        if web_snippets_list and web_strong:
            return _finalize_answer(
                _gen_web("No strong match in your documents; answer uses web results."),
                routing="web_weak_docs",
                retrieval_ran=True,
                retrieval_hit_count=len(hits),
                fallback_to_general=True,
                **common_diag,
            )
        if web_snippets_list:
            if user_expects_document_grounding(query):
                text, gerr = safe_document_abstain_answer(query)
            else:
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
        text, gerr = _general_or_abstain(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=MSG_GROUNDED_FALLBACK_NOTE),
            routing="route_generation_failed",
            retrieval_ran=True,
            retrieval_hit_count=len(hits),
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )

    doc_scope = user_expects_document_grounding(query)
    if stream:
        _stream_fn = stream_document_abstain_tokens if doc_scope else stream_general_answer_tokens

        return _finalize_answer(
            AssistantTurn(
                mode="general",
                text="",
                assistant_note="No strong document match and no web results; general answer only.",
                stream_tokens=lambda: _stream_fn(query),
            ),
            routing="general_weak_no_web",
            retrieval_ran=True,
            retrieval_hit_count=len(hits),
            fallback_to_general=True,
            exception_summary=None,
        )
    if doc_scope:
        text, gerr = safe_document_abstain_answer(query)
    else:
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
        text, gerr = _general_or_abstain(query)
        return _finalize_answer(
            AssistantTurn(mode="general", text=text, assistant_note=None),
            routing="general_unhandled_fallback",
            retrieval_ran=False,
            retrieval_hit_count=0,
            fallback_to_general=True,
            exception_summary=gerr or debug_service.short_exc(exc),
        )
