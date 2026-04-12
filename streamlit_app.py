"""Streamlit UI for Knowledge Assistant. Run: streamlit run streamlit_app.py"""

from __future__ import annotations

import hashlib
import html
import time
from pathlib import Path
from typing import Any

import streamlit as st

from app import streamlit_icons as st_icons
from app.ingestion.loader import get_default_raw_dir
from app.persistence import chat_store, document_manifest
from app.retrieval.vector_store import get_default_faiss_folder
from app.services import chat_service, debug_service, index_service, message_service, perf_service, upload_service
from app.services.library_delete import delete_library_document

APP_NAME = message_service.APP_NAME
EMPTY_STATE_VALUE_PROP = message_service.EMPTY_STATE_VALUE_PROP
HERO_BEST_FOR = message_service.HERO_BEST_FOR
SIDEBAR_CAPTION = message_service.SIDEBAR_CAPTION
STARTER_QUESTIONS = message_service.STARTER_QUESTIONS
WHY_THIS_WORKSPACE_MD = message_service.WHY_THIS_WORKSPACE_MD
COMPARISON_MD = message_service.COMPARISON_MD
DEFAULT_TOP_K = message_service.DEFAULT_TOP_K
SIDEBAR_TOP_K_MAX = message_service.SIDEBAR_TOP_K_MAX
MSG_LIBRARY_UPDATED = message_service.MSG_LIBRARY_UPDATED
MSG_UPLOAD_FAILED = message_service.MSG_UPLOAD_FAILED
MSG_DOCS_PREP_FAILED = message_service.MSG_DOCS_PREP_FAILED

# Compact task mode labels (Preferences expander); values are service-layer mode keys.
_TASK_MODE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Auto"),
    ("summarize", "Summarize"),
    ("extract", "Extract"),
    ("compare", "Compare"),
]

# Short labels for the subtle line above the chat input (demo clarity without clutter).
_TASK_MODE_COMPOSER_HINT: dict[str, str] = {
    "auto": "Chat & docs",
    "summarize": "Summarize",
    "extract": "Extract",
    "compare": "Compare",
}

# Same widget key pattern as before: keeps ingest + sync behavior identical.
_COMPOSER_KEY_PREFIX = "composer_"


def ingest_composer_attachments(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str | None, bool]:
    """
    Save files from the current composer widget and rebuild the library index.

    Returns (optional_warning_for_status_line, library_index_refreshed_ok).
    Warnings are non-fatal: callers should still run the assistant for the user's question.
    """
    rid = int(st.session_state.get("composer_reset", 0))
    key = f"{_COMPOSER_KEY_PREFIX}{rid}"
    uploaded = st.session_state.get(key)
    if debug_service.debug_enabled():
        detected: list[str] = []
        if uploaded:
            for f in uploaded:
                detected.append(getattr(f, "name", str(f)))
        debug_service.merge(composer_widget_key=key, uploaded_detected=detected)
    try:
        n, saved_names = upload_service.save_uploads_to_raw(uploaded, raw_dir)
    except OSError as exc:
        if debug_service.debug_enabled():
            debug_service.merge(
                saved_filenames=[],
                ingest_rebuild="skipped_save_failed",
                exception_summary=debug_service.short_exc(exc),
            )
        return MSG_UPLOAD_FAILED, False
    if debug_service.debug_enabled():
        debug_service.merge(saved_filenames=saved_names)
    if n == 0:
        if debug_service.debug_enabled():
            debug_service.merge(ingest_rebuild="skipped_no_new_files")
        return None, False
    ok, err_msg, _, _ = index_service.rebuild_knowledge_index(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if ok:
        st.session_state.kb_sync_fingerprint = index_service.library_content_fingerprint(raw_dir)
        st.session_state.composer_reset = rid + 1
        if debug_service.debug_enabled():
            debug_service.merge(ingest_rebuild="rebuilt")
        return None, True
    if debug_service.debug_enabled():
        debug_service.merge(ingest_rebuild="failed", ingest_error_summary=str(err_msg)[:200])
    return MSG_DOCS_PREP_FAILED, False


def sync_documents_manual(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Save any pending uploads, rebuild search data from disk, refresh sync fingerprint (UI entry point)."""
    rid = int(st.session_state.get("composer_reset", 0))
    key = f"{_COMPOSER_KEY_PREFIX}{rid}"
    uploaded = st.session_state.get(key)
    if debug_service.debug_enabled():
        detected: list[str] = []
        if uploaded:
            for f in uploaded:
                detected.append(getattr(f, "name", str(f)))
        debug_service.merge(manual_sync_widget_key=key, manual_sync_uploads_detected=detected)
    try:
        n, saved_names = upload_service.save_uploads_to_raw(uploaded, raw_dir)
    except OSError as exc:
        if debug_service.debug_enabled():
            debug_service.merge(
                manual_sync_save_failed=True,
                exception_summary=debug_service.short_exc(exc),
            )
        st.warning("Couldn't save your files. Check permissions and try again.")
        return
    if debug_service.debug_enabled():
        debug_service.merge(manual_sync_saved_filenames=saved_names)
    ok, err_msg, _, _ = index_service.rebuild_knowledge_index(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if ok:
        st.session_state.kb_sync_fingerprint = index_service.library_content_fingerprint(raw_dir)
        if n > 0:
            st.session_state.composer_reset = rid + 1
        if debug_service.debug_enabled():
            debug_service.merge(manual_sync="rebuilt_ok")
        st.session_state["_ka_sync_toast"] = True
        return
    if debug_service.debug_enabled():
        debug_service.merge(manual_sync="rebuild_failed", manual_sync_error_summary=str(err_msg)[:200])
    st.warning(err_msg or "Couldn't sync documents. Try again.")


def read_settings() -> tuple[int, int, int]:
    cs = int(st.session_state.get("settings_chunk_size", st.session_state.get("adv_chunk_size", 900)))
    co = int(st.session_state.get("settings_chunk_overlap", st.session_state.get("adv_chunk_overlap", 120)))
    tk = int(st.session_state.get("settings_top_k", st.session_state.get("adv_top_k", DEFAULT_TOP_K)))
    tk = max(1, min(SIDEBAR_TOP_K_MAX, tk))
    return cs, co, tk


def init_session() -> None:
    chat_store.init_db()
    if "active_chat_id" not in st.session_state or not st.session_state.active_chat_id:
        st.session_state.active_chat_id = chat_store.create_session()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_sync_fingerprint" not in st.session_state:
        st.session_state.kb_sync_fingerprint = None
    if "composer_reset" not in st.session_state:
        st.session_state.composer_reset = 0
    if "settings_chunk_size" not in st.session_state and "adv_chunk_size" in st.session_state:
        st.session_state.settings_chunk_size = st.session_state["adv_chunk_size"]
    if "settings_chunk_overlap" not in st.session_state and "adv_chunk_overlap" in st.session_state:
        st.session_state.settings_chunk_overlap = st.session_state["adv_chunk_overlap"]
    if "settings_top_k" not in st.session_state and "adv_top_k" in st.session_state:
        st.session_state.settings_top_k = st.session_state["adv_top_k"]
    _tk0 = int(st.session_state.get("settings_top_k", DEFAULT_TOP_K))
    if _tk0 > SIDEBAR_TOP_K_MAX or _tk0 < 1:
        st.session_state.settings_top_k = max(1, min(SIDEBAR_TOP_K_MAX, _tk0))
    if "task_mode" not in st.session_state:
        st.session_state.task_mode = "auto"
    if "summarize_scope" not in st.session_state:
        st.session_state.summarize_scope = "all"


def read_task_settings() -> tuple[str, str]:
    """Active document task mode and summarize scope (filename or ``all``)."""
    mode = str(st.session_state.get("task_mode", "auto"))
    if mode not in ("auto", "summarize", "extract", "compare"):
        mode = "auto"
    scope = str(st.session_state.get("summarize_scope", "all"))
    return mode, scope


def new_chat() -> None:
    """Start a new persisted chat session; library on disk unchanged."""
    st.session_state.active_chat_id = chat_store.create_session()
    st.session_state.messages = chat_store.load_messages(st.session_state.active_chat_id)


def reload_messages_from_store() -> None:
    st.session_state.messages = chat_store.load_messages(st.session_state.active_chat_id)


def persist_assistant_turn_db(
    turn: chat_service.AssistantTurn,
    *,
    ingest_note: str | None = None,
) -> None:
    """Write assistant turn to SQLite (mirrors append_assistant_turn payload shape)."""
    sid = st.session_state.active_chat_id
    note = message_service.merge_notes(ingest_note, turn.assistant_note)
    if turn.mode == "error":
        content = turn.error or turn.text
        extra: dict[str, Any] = {}
    elif turn.mode == "general":
        content = turn.text
        extra = {}
    elif turn.mode == "web":
        content = turn.text
        extra = {"web_only": True}
        if turn.web_snippets:
            extra["web_sources"] = turn.web_snippets
    elif turn.mode == "blended" and turn.grounded:
        pl = chat_service.assistant_payload(turn.grounded, turn.hits)
        content = pl["content"]
        extra = {k: v for k, v in pl.items() if k != "content"}
        extra["blended"] = True
        if turn.web_snippets:
            extra["web_sources"] = turn.web_snippets
    elif turn.mode == "grounded" and turn.grounded:
        pl = chat_service.assistant_payload(turn.grounded, turn.hits)
        content = pl["content"]
        extra = {k: v for k, v in pl.items() if k != "content"}
    else:
        content = turn.text
        extra = {}
    if note:
        extra["status_note"] = note
    chat_store.append_message(sid, "assistant", content, extra)


def maybe_set_session_title_from_first_user_message(session_id: str, user_text: str) -> None:
    msgs = chat_store.load_messages(session_id)
    if len(msgs) == 1 and msgs[0].get("role") == "user":
        chat_store.set_session_title(session_id, user_text.strip()[:80])


def turn_to_ui_message(turn: chat_service.AssistantTurn) -> dict[str, Any]:
    """Shape assistant turn for :func:`render_assistant_message` (non-streaming)."""
    if turn.mode == "error":
        return {"role": "assistant", "content": turn.error or turn.text}
    msg: dict[str, Any] = {"role": "assistant", "content": turn.text}
    if turn.assistant_note:
        msg["status_note"] = turn.assistant_note
    if turn.mode == "grounded" and turn.grounded:
        pl = chat_service.assistant_payload(turn.grounded, turn.hits)
        msg["grounded"] = pl.get("grounded")
        msg["sources"] = pl.get("sources")
        msg["excerpts"] = pl.get("excerpts")
    elif turn.mode == "web":
        msg["web_only"] = True
        if turn.web_snippets:
            msg["web_sources"] = turn.web_snippets
    elif turn.mode == "blended" and turn.grounded:
        pl = chat_service.assistant_payload(turn.grounded, turn.hits)
        msg["grounded"] = pl.get("grounded")
        msg["sources"] = pl.get("sources")
        msg["excerpts"] = pl.get("excerpts")
        msg["blended"] = True
        if turn.web_snippets:
            msg["web_sources"] = turn.web_snippets
    return msg


def is_grounded_assistant_message(msg: dict[str, Any]) -> bool:
    """Document-grounded turns (Sources / excerpts)."""
    if msg.get("grounded"):
        return True
    if msg.get("sources") or msg.get("excerpts") or msg.get("retrieved_passages"):
        return True
    return False


def has_web_sources(msg: dict[str, Any]) -> bool:
    return bool(msg.get("web_sources"))


def _mode_chip_label_from_turn(turn: chat_service.AssistantTurn) -> str | None:
    if turn.mode == "web":
        return "Web"
    if turn.mode == "blended":
        return "Documents + web"
    if turn.mode == "grounded":
        return "Documents"
    return None


def _mode_chip_label_from_message(msg: dict[str, Any]) -> str | None:
    if msg.get("blended"):
        return "Documents + web"
    if msg.get("web_only"):
        return "Web"
    if is_grounded_assistant_message(msg):
        return "Documents"
    return None


def render_answer_mode_chip(label: str | None) -> None:
    if not label:
        return
    st.markdown(
        f'<p class="ka-answer-chip"><span class="ka-answer-chip-inner">{html.escape(label)}</span></p>',
        unsafe_allow_html=True,
    )


def run_pending_assistant_turn(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    cs: int,
    co: int,
    tk: int,
    task_mode: str,
    summarize_scope: str,
) -> None:
    pend = st.session_state.pop("_pending_assistant", None)
    if not pend:
        return
    t0 = time.perf_counter()
    debug_service.debug_begin_turn(str(pend.get("turn_source", "chat_input")))
    if debug_service.debug_enabled():
        debug_service.merge(
            task_mode=task_mode,
            summarize_scope=summarize_scope,
            pending_query_preview=str(pend.get("q", ""))[:120],
        )
    with st.chat_message("assistant"):
        with st.spinner("Preparing your answer…"):
            try:
                msgs = list(st.session_state.messages)
                qstrip = str(pend["q"]).strip()
                conv_hist = (
                    msgs[:-1]
                    if msgs
                    and str(msgs[-1].get("role") or "") == "user"
                    and str(msgs[-1].get("content") or "").strip() == qstrip
                    else msgs
                )
                turn = chat_service.answer_user_query(
                    str(pend["q"]),
                    raw_dir=raw_dir,
                    faiss_folder=faiss_folder,
                    chunk_size=int(pend.get("cs", cs)),
                    chunk_overlap=int(pend.get("co", co)),
                    top_k=int(pend.get("tk", tk)),
                    task_mode=str(pend.get("task_mode", task_mode)),
                    summarize_scope=str(pend.get("summarize_scope", summarize_scope)),
                    conversation_history=conv_hist,
                )
            except Exception as exc:
                debug_service.merge(exception_summary=debug_service.short_exc(exc))
                text, _gerr = chat_service.safe_general_answer(str(pend["q"]))
                turn = chat_service.AssistantTurn(mode="general", text=text)
        ingest_note = message_service.merge_notes(
            pend.get("ingest_warn"),
            MSG_LIBRARY_UPDATED if pend.get("library_refreshed") else None,
        )
        if turn.stream_tokens:
            render_answer_mode_chip(_mode_chip_label_from_turn(turn))
            first = True
            t_stream = time.perf_counter()

            def gen():
                nonlocal first
                for piece in turn.stream_tokens():
                    if first:
                        perf_service.record_phase_ms("ttft", (time.perf_counter() - t0) * 1000)
                        first = False
                    yield piece

            full = st.write_stream(gen())
            perf_service.record_phase_ms("generation", (time.perf_counter() - t_stream) * 1000)
            final_turn = chat_service.materialize_streamed_turn(turn, full)
        else:
            final_turn = turn
            render_assistant_message(turn_to_ui_message(turn))
    persist_assistant_turn_db(final_turn, ingest_note=ingest_note)
    dbg = dict(st.session_state.get("_dev_debug", {}))
    perf_service.log_answer_pipeline_metrics(
        routing=str(dbg.get("routing", "unknown")),
        ttft_ms=dbg.get("latency_ms_ttft"),
        total_ms=round((time.perf_counter() - t0) * 1000, 2),
        retrieval_ms=dbg.get("latency_ms_retrieval"),
        generation_ms=dbg.get("latency_ms_generation"),
        rewrite_ms=dbg.get("latency_ms_rewrite"),
    )
    if pend.get("library_refreshed"):
        st.session_state["_ka_ingest_toast"] = True
    reload_messages_from_store()
    st.rerun()


def render_user_message(content: str) -> None:
    st.markdown(content)


def render_sources_expander(sources: list[dict[str, Any]]) -> None:
    with st.expander("Document sources", expanded=False):
        for i, s in enumerate(sources):
            if i:
                st.markdown('<div class="ka-src-gap"></div>', unsafe_allow_html=True)
            sn = s.get("source_number", "?")
            label = str(s.get("source_label") or s.get("source_name", "") or "-")
            st.markdown(
                f'<p class="ka-src-line"><span class="ka-src-num">[{html.escape(str(sn))}]</span> '
                f'<span class="ka-src-name">{html.escape(label)}</span></p>',
                unsafe_allow_html=True,
            )
            snip = html.escape(str(s.get("snippet") or "")[:360])
            if snip:
                st.markdown(f'<div class="ka-quote">{snip}</div>', unsafe_allow_html=True)
            fp = message_service.display_path_hint(s.get("file_path"))
            if fp:
                st.caption(fp)


def render_web_sources_expander(web_sources: list[dict[str, Any]]) -> None:
    with st.expander("Web sources · [WEB n] in reply", expanded=False):
        for i, w in enumerate(web_sources):
            if i:
                st.markdown('<div class="ka-src-gap"></div>', unsafe_allow_html=True)
            ref = str(w.get("web_ref") or f"WEB {i + 1}")
            st.caption(ref)
            title = str(w.get("title") or "Link")
            url = str(w.get("url") or "").strip()
            if url:
                st.markdown(f"**[{title}]({url})**")
            else:
                st.markdown(f"**{title}**")
            snip = html.escape(str(w.get("snippet") or "")[:500])
            if snip:
                st.markdown(f'<div class="ka-web-snip">{snip}</div>', unsafe_allow_html=True)


def render_excerpts_expander(excerpts: list[dict[str, Any]]) -> None:
    with st.expander("Quotes from your files", expanded=False):
        for i, p in enumerate(excerpts):
            if i:
                st.markdown('<div class="ka-ex-gap"></div>', unsafe_allow_html=True)
            fn = p.get("file_name", "")
            pg = p.get("page_label", "-")
            st.markdown(
                f'<p class="ka-ex-head">{html.escape(str(fn))} <span class="ka-ex-page">· p. {html.escape(str(pg))}</span></p>',
                unsafe_allow_html=True,
            )
            body = html.escape(str(p.get("preview_text") or "").strip() or "-")
            st.markdown(f'<div class="ka-quote">{body}</div>', unsafe_allow_html=True)


def render_composer_task_hint(task_mode: str, summarize_scope: str) -> None:
    """Muted one-liner above the chat input (current task; hidden for default Auto)."""
    if task_mode == "auto":
        return
    label = _TASK_MODE_COMPOSER_HINT.get(task_mode)
    if not label:
        return
    extra = ""
    if task_mode == "summarize" and summarize_scope and summarize_scope != "all":
        extra = f" · {html.escape(summarize_scope)}"
    st.markdown(
        f'<p class="ka-composer-hint"><span class="ka-composer-hint-k">Task</span> {html.escape(label)}{extra}</p>',
        unsafe_allow_html=True,
    )


def render_grounded_expanders(msg: dict[str, Any]) -> None:
    sources = msg.get("sources") or []
    legacy_passages = msg.get("retrieved_passages") or []
    excerpts = msg.get("excerpts") or legacy_passages
    if sources:
        render_sources_expander(sources)
    if excerpts:
        render_excerpts_expander(excerpts)


def render_empty_state_hero() -> None:
    """Title, value prop, best-for context, and starter prompt entry (no messages yet)."""
    st.markdown(
        f"""
        <div class="ka-hero-wrap">
          <div class="ka-hero-title">{html.escape(APP_NAME)}</div>
          <p class="ka-value">{html.escape(EMPTY_STATE_VALUE_PROP)}</p>
          <p class="ka-best-for">{html.escape(HERO_BEST_FOR)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="ka-empty-label">Suggested</p>', unsafe_allow_html=True)
    for i, q in enumerate(STARTER_QUESTIONS):
        if st.button(q, key=f"starter_{i}", use_container_width=True, type="secondary"):
            st.session_state.pending_starter = q
            st.rerun()
    st.markdown('<div class="ka-empty-pad"></div>', unsafe_allow_html=True)


def render_sidebar_positioning() -> None:
    """Trust and positioning copy: why this workspace, factual comparison (sidebar, collapsed by default)."""
    with st.expander("Why this workspace", expanded=False):
        st.markdown(WHY_THIS_WORKSPACE_MD.strip())
    with st.expander("How this compares", expanded=False):
        st.caption("Factual differences in focus, not a ranking of quality.")
        st.markdown(COMPARISON_MD.strip())


def render_assistant_message(msg: dict[str, Any]) -> None:
    render_answer_mode_chip(_mode_chip_label_from_message(msg))
    st.markdown(msg.get("content") or "")
    if msg.get("status_note"):
        st.markdown(
            f'<p class="ka-msg-status">{message_service.render_status_note_html(str(msg["status_note"]))}</p>',
            unsafe_allow_html=True,
        )
    if is_grounded_assistant_message(msg):
        render_grounded_expanders(msg)
    if has_web_sources(msg):
        render_web_sources_expander(msg["web_sources"])


def render_sidebar_chats() -> None:
    """GPT-style chat list + switcher."""
    st.markdown('<div class="ka-side-divider" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.markdown('<p class="ka-side-h">Chats</p>', unsafe_allow_html=True)
    cur = st.session_state.active_chat_id
    row = chat_store.get_session(cur)
    if row is None:
        st.session_state.active_chat_id = chat_store.create_session()
        reload_messages_from_store()
        st.rerun()
    sessions = chat_store.list_sessions(50)
    ids = [s["id"] for s in sessions]
    titles = {s["id"]: (s["title"] or "Chat")[:44] for s in sessions}
    if cur not in ids:
        titles[cur] = (row["title"] if row else "Chat")[:44]
        ids = [cur] + [i for i in ids if i != cur]
    idx = ids.index(cur) if cur in ids else 0
    picked = st.selectbox(
        "Open chat",
        options=ids,
        index=idx,
        format_func=lambda i: titles.get(i, i[:8]),
        label_visibility="collapsed",
    )
    if picked != cur:
        st.session_state.active_chat_id = picked
        reload_messages_from_store()
        st.rerun()
    if st.button("New chat", key="ka_new_chat_sidebar", use_container_width=True, type="secondary"):
        new_chat()
        st.rerun()
    if st.button(
        "Delete this chat",
        key="ka_delete_this_chat",
        use_container_width=True,
        help="Remove this conversation from history",
    ):
        chat_store.delete_session(cur)
        nxt = chat_store.list_sessions(50)
        st.session_state.active_chat_id = (
            nxt[0]["id"] if nxt else chat_store.create_session()
        )
        reload_messages_from_store()
        st.rerun()
    st.markdown('<div class="ka-side-sp"></div>', unsafe_allow_html=True)


def _doc_status_badge(st_: str, row: dict[str, Any]) -> str:
    if st_ == "ready_limited" or (st_ == "ready" and row.get("retrieval_quality") == "weak"):
        return "Ready · limited"
    labels = {
        "ready": "Ready",
        "ready_limited": "Ready · limited",
        "processing": "Indexing…",
        "failed": "Needs attention",
        "uploaded": "Pending sync",
    }
    return labels.get(st_, st_)


def render_sidebar_documents_and_actions(
    raw_dir: Path,
    faiss_folder: Path,
) -> None:
    """Upload, library list, preferences (before sync so values apply), sync, new chat, positioning, debug."""
    st.markdown('<p class="ka-side-h">Upload</p>', unsafe_allow_html=True)
    st.caption("PDF, Word, or plain text — attach here, then Sync or send a message.")
    rid = int(st.session_state.get("composer_reset", 0))
    st.file_uploader(
        "Add files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"{_COMPOSER_KEY_PREFIX}{rid}",
        label_visibility="visible",
        help="Files are saved to your local library folder. Sync builds the search index.",
    )

    st.markdown('<p class="ka-side-h">Library</p>', unsafe_allow_html=True)
    files_now = index_service.list_raw_files(raw_dir)
    lib_manifest = document_manifest.load_manifest(faiss_folder)
    files_meta = lib_manifest.get("files") or {}
    if files_now:
        for p in files_now:
            row = dict(files_meta.get(p.name) or {})
            st_ = str(
                row.get("status") or document_manifest.file_health_status(faiss_folder, p.name)
            )
            badge = html.escape(_doc_status_badge(st_, row))
            fc1, fc2 = st.columns([5, 1])
            with fc1:
                st.markdown(
                    f'<p class="ka-lib-file">{html.escape(p.name)} '
                    f'<span class="ka-lib-status">({badge})</span></p>',
                    unsafe_allow_html=True,
                )
            with fc2:
                del_key = f"ka_del_{hashlib.md5(p.name.encode('utf-8')).hexdigest()[:16]}"
                if st.button("×", key=del_key, help=f"Remove {p.name} from library"):
                    cs, co, _tk = read_settings()
                    ok, msg = delete_library_document(
                        raw_dir, faiss_folder, p.name, chunk_size=cs, chunk_overlap=co
                    )
                    if ok:
                        st.session_state.kb_sync_fingerprint = (
                            index_service.library_content_fingerprint(raw_dir)
                        )
                    st.toast(
                        (msg or ("Removed" if ok else "Could not remove"))[:200],
                        icon=st_icons.TOAST_DELETE if ok else st_icons.TOAST_ERROR,
                    )
                    st.rerun()
            note = row.get("user_facing_note")
            if not note and row.get("retrieval_quality_note") and row.get("retrieval_quality") == "weak":
                note = row.get("retrieval_quality_note")
            if note:
                st.caption(html.escape(str(note)[:200]))
        hc = document_manifest.library_health_counts(faiss_folder)
        if any(hc.get(k, 0) for k in ("ready", "ready_limited", "failed")):
            st.caption(
                html.escape(
                    f"{hc.get('ready', 0)} ready · {hc.get('ready_limited', 0)} limited · "
                    f"{hc.get('failed', 0)} need attention"
                )
            )
    else:
        st.markdown(
            '<p class="ka-lib-empty">No files yet. Upload, then Sync.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ka-side-sp"></div>', unsafe_allow_html=True)

    # Render before Sync so chunk / top_k values are current for this run when the button fires.
    with st.expander("Preferences", expanded=False):
        _tk = max(1, min(SIDEBAR_TOP_K_MAX, int(st.session_state.get("settings_top_k", DEFAULT_TOP_K))))
        st.session_state.settings_top_k = st.select_slider(
            "Sources to show",
            options=list(range(1, SIDEBAR_TOP_K_MAX + 1)),
            value=_tk,
            help="How many document passages can inform one answer.",
        )
        st.session_state.settings_chunk_size = st.number_input(
            "Passage size",
            100,
            4000,
            int(st.session_state.get("settings_chunk_size", 900)),
            step=50,
            help="Larger passages carry more context per match. Adjust only if needed.",
        )
        st.session_state.settings_chunk_overlap = st.number_input(
            "Passage overlap",
            0,
            2000,
            int(st.session_state.get("settings_chunk_overlap", 120)),
            step=10,
            help="Shared text between passages. Usually leave as-is.",
        )
        st.markdown('<div class="ka-side-sp"></div>', unsafe_allow_html=True)
        _vals = [v for v, _ in _TASK_MODE_OPTIONS]
        _labels = {v: lab for v, lab in _TASK_MODE_OPTIONS}
        _cur_m = st.session_state.get("task_mode", "auto")
        if _cur_m not in _vals:
            _cur_m = "auto"
        _mi = _vals.index(_cur_m)
        st.session_state.task_mode = st.selectbox(
            "Task mode",
            options=_vals,
            index=_mi,
            format_func=lambda k: _labels.get(str(k), str(k)),
            help="Auto: normal chat and Q&A. Other modes use a focused prompt on your synced library.",
        )
        if st.session_state.task_mode == "summarize":
            _files = index_service.list_raw_files(raw_dir)
            _scope_opts = ["all"] + [p.name for p in _files]
            _sc = st.session_state.get("summarize_scope", "all")
            if _sc not in _scope_opts:
                _sc = "all"
            st.session_state.summarize_scope = st.selectbox(
                "Summarize scope",
                options=_scope_opts,
                index=_scope_opts.index(_sc),
                format_func=lambda x: "All documents" if x == "all" else str(x),
                help="All documents or one file from your library.",
            )

    cs, co, _tk_read = read_settings()
    if st.button("Sync library", use_container_width=True, type="primary"):
        sync_documents_manual(
            raw_dir,
            faiss_folder,
            chunk_size=cs,
            chunk_overlap=co,
        )
        st.rerun()

    st.markdown('<div class="ka-side-sp"></div>', unsafe_allow_html=True)
    render_sidebar_positioning()

    debug_service.render_debug_panel()


# set_page_config must run before any other Streamlit command (including session_state).
st.set_page_config(
    page_title=APP_NAME,
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

init_session()
reload_messages_from_store()

raw_dir = get_default_raw_dir()
raw_dir.mkdir(parents=True, exist_ok=True)
faiss_folder = get_default_faiss_folder()

st.markdown(
    """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
    :root {
      --ka-text: rgba(15, 23, 42, 0.94);
      --ka-muted: rgba(15, 23, 42, 0.5);
      --ka-line: rgba(15, 23, 42, 0.085);
      --ka-sidebar: rgba(252, 252, 253, 0.98);
      --ka-surface: rgba(255, 255, 255, 0.82);
      --ka-chat-user: rgba(239, 246, 255, 0.92);
      --ka-chat-asst: rgba(255, 255, 255, 0.96);
      --ka-accent: rgba(37, 99, 235, 0.42);
      --ka-app-bg-top: #f8fafc;
      --ka-app-bg-mid: #f1f5f9;
      --ka-app-bg-bot: #eceff3;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --ka-text: rgba(248, 250, 252, 0.94);
        --ka-muted: rgba(248, 250, 252, 0.52);
        --ka-line: rgba(255, 255, 255, 0.1);
        --ka-sidebar: rgba(15, 23, 42, 0.99);
        --ka-surface: rgba(30, 41, 59, 0.58);
        --ka-chat-user: rgba(30, 58, 95, 0.55);
        --ka-chat-asst: rgba(15, 23, 42, 0.62);
        --ka-accent: rgba(96, 165, 250, 0.5);
        --ka-app-bg-top: #0f172a;
        --ka-app-bg-mid: #111827;
        --ka-app-bg-bot: #0b1220;
      }
    }
    .stApp {
      font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      background: linear-gradient(
        168deg,
        var(--ka-app-bg-top) 0%,
        var(--ka-app-bg-mid) 38%,
        var(--ka-app-bg-bot) 100%
      ) !important;
      color: var(--ka-text);
    }
    #MainMenu { visibility: hidden; }
    footer[data-testid="stFooter"] { display: none !important; }
    header[data-testid="stHeader"] {
      background: transparent !important;
      backdrop-filter: none !important;
      border-bottom: none !important;
    }
    [data-testid="stDecoration"] { display: none !important; }
    .main .block-container {
      padding-top: 1rem !important;
      /* Flush chat input to viewport bottom (Streamlit adds extra block padding by default). */
      padding-bottom: 0 !important;
      max-width: 52rem !important;
      margin-left: auto !important;
      margin-right: auto !important;
    }
    .ka-composer-hint {
      font-size: 0.72rem !important;
      line-height: 1.4 !important;
      color: var(--ka-muted) !important;
      margin: 0 0 0.35rem 0 !important;
      padding: 0 0.15rem;
      letter-spacing: 0.02em;
    }
    .ka-composer-hint-k {
      text-transform: uppercase;
      font-weight: 600;
      font-size: 0.62rem;
      letter-spacing: 0.08em;
      margin-right: 0.35rem;
      opacity: 0.85;
    }
    /* Wrappers around the composer often retain default block spacing. */
    .main [data-testid="element-container"]:has([data-testid="stChatInput"]) {
      margin-bottom: 0 !important;
      padding-bottom: 0 !important;
    }
    /* Inner Emotion layout shell (direct child of element-container): still carried bottom margin/padding. */
    .main [data-testid="element-container"]:has([data-testid="stChatInput"]) > div {
      margin-bottom: 0 !important;
      padding-bottom: 0 !important;
    }
    /* Main column root under .block-container — only the subtree that contains the composer. */
    .main .block-container > div:has([data-testid="stChatInput"]) {
      margin-bottom: 0 !important;
      padding-bottom: 0 !important;
    }
    /* Flex column that stacks the markdown hint + chat input (theme padding on the block itself). */
    .main [data-testid="stVerticalBlock"]:has([data-testid="stChatInput"]) {
      margin-bottom: 0 !important;
      padding-bottom: 0 !important;
    }
    .main [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stChatInput"]) {
      margin-bottom: 0 !important;
      padding-bottom: 0 !important;
    }
    div[data-testid="stChatInput"],
    div[data-testid="stChatInputContainer"],
    .stChatInputContainer {
      padding-bottom: 0 !important;
      margin-bottom: 0 !important;
    }
    div[data-testid="stChatInput"] {
      padding-top: 0.75rem;
      border-top: 1px solid var(--ka-line);
      margin-top: 0.35rem;
      background: linear-gradient(180deg, transparent 0%, var(--ka-app-bg-bot) 35%);
    }
    div[data-testid="stChatInput"] textarea {
      font-size: 0.97rem !important;
      line-height: 1.5 !important;
    }
    /* Direct child shell + textarea parent: theme still left bottom padding/margin on these nodes. */
    .main div[data-testid="stChatInput"] > div:has([data-testid="stChatInputTextArea"]),
    .main div[data-testid="stChatInput"] div:has(> [data-testid="stChatInputTextArea"]) {
      padding-bottom: 0 !important;
      margin-bottom: 0 !important;
    }
    [data-testid="stChatMessage"] {
      padding: 0.75rem 1rem 0.85rem 1rem !important;
      margin-bottom: 1.35rem !important;
      border-radius: 1rem;
      border: 1px solid var(--ka-line);
      background: var(--ka-chat-asst);
      box-shadow: 0 2px 14px rgba(15, 23, 42, 0.055);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
      background: var(--ka-chat-user);
      border-left: 3px solid var(--ka-accent);
      box-shadow: 0 2px 12px rgba(37, 99, 235, 0.07);
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
      line-height: 1.68;
      font-size: 0.96rem;
      margin-bottom: 0.52em;
      color: var(--ka-text);
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
      line-height: 1.52;
      font-size: 0.86rem;
      border-radius: 0.45rem;
      border: 1px solid var(--ka-line);
      background: var(--ka-surface) !important;
      padding: 0.55rem 0.65rem;
    }
    p.ka-msg-status {
      margin: 0.75rem 0 0 0 !important;
      font-size: 0.72rem !important;
      line-height: 1.5 !important;
      color: var(--ka-muted) !important;
      letter-spacing: 0.015em;
    }
    p.ka-answer-chip {
      margin: 0 0 0.55rem 0 !important;
      line-height: 1 !important;
    }
    .ka-answer-chip-inner {
      display: inline-block;
      font-size: 0.62rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ka-muted);
      padding: 0.22rem 0.55rem;
      border-radius: 999px;
      border: 1px solid var(--ka-line);
      background: var(--ka-surface);
    }
    div.ka-quote {
      margin: 0 !important;
      padding: 0.62rem 0.72rem !important;
      font-size: 0.835rem !important;
      line-height: 1.55 !important;
      white-space: pre-wrap !important;
      word-break: break-word;
      border-radius: 0.5rem;
      border: 1px solid var(--ka-line);
      background: var(--ka-surface) !important;
      color: var(--ka-text) !important;
    }
    div.ka-web-snip {
      margin-top: 0.3rem;
      margin-bottom: 0.1rem;
      padding: 0.5rem 0.62rem;
      font-size: 0.8rem;
      line-height: 1.48;
      color: var(--ka-text);
      opacity: 0.9;
      border-radius: 0.45rem;
      background: var(--ka-surface);
      border: 1px solid var(--ka-line);
    }
    [data-testid="stChatMessage"] [data-testid="stSpinner"] > div {
      border-color: var(--ka-accent) transparent transparent transparent !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
      margin-top: 0.55rem !important;
    }
    [data-testid="stExpander"] {
      margin-top: 0.5rem !important;
      margin-bottom: 0.2rem !important;
    }
    [data-testid="stExpander"] details {
      border: 1px solid var(--ka-line) !important;
      border-radius: 0.5rem !important;
      background: var(--ka-surface) !important;
    }
    [data-testid="stExpander"] summary {
      font-weight: 500;
      font-size: 0.82rem;
      letter-spacing: 0.02em;
      padding: 0.5rem 0.7rem;
      color: var(--ka-text);
      opacity: 0.92;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
      margin-bottom: 0.4rem;
      font-size: 0.86rem;
      line-height: 1.5;
    }
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
      margin-top: 0.15rem;
      opacity: 0.88;
      font-size: 0.78rem !important;
    }
    p.ka-src-line {
      margin: 0 !important;
      line-height: 1.45 !important;
      font-size: 0.86rem !important;
      color: var(--ka-text) !important;
    }
    .ka-src-num { font-weight: 600; opacity: 0.9; margin-right: 0.25rem; }
    .ka-src-name { font-weight: 500; }
    .ka-src-meta { color: var(--ka-muted); font-size: 0.82rem; }
    p.ka-ex-head {
      margin: 0 0 0.35rem 0 !important;
      font-size: 0.82rem !important;
      font-weight: 600;
      color: var(--ka-text) !important;
    }
    .ka-ex-page { font-weight: 400; color: var(--ka-muted); }
    pre.ka-ex-body {
      margin: 0 !important;
      padding: 0.55rem 0.65rem !important;
      font-size: 0.8rem !important;
      line-height: 1.48 !important;
      white-space: pre-wrap !important;
      word-break: break-word;
      border-radius: 0.45rem;
      border: 1px solid var(--ka-line);
      background: var(--ka-surface) !important;
      color: var(--ka-text) !important;
      font-family: ui-sans-serif, system-ui, sans-serif !important;
    }
    .ka-src-gap { height: 0.55rem; }
    .ka-ex-gap {
      height: 0.6rem;
      margin: 0.5rem 0;
      border-top: 1px solid var(--ka-line);
    }
    [data-testid="stSidebar"] {
      background: var(--ka-sidebar) !important;
      border-right: 1px solid var(--ka-line) !important;
    }
    [data-testid="stSidebarContent"] {
      padding-top: 1.1rem !important;
      padding-bottom: 1.35rem !important;
      padding-left: 0.9rem !important;
      padding-right: 0.9rem !important;
      font-size: 0.89rem !important;
    }
    .ka-brand-title {
      font-weight: 700;
      font-size: 1.05rem;
      letter-spacing: -0.04em;
      color: var(--ka-text);
      margin: 0 0 0.35rem 0;
      line-height: 1.2;
    }
    .ka-brand-sub {
      font-size: 0.78rem;
      line-height: 1.52;
      color: var(--ka-muted);
      margin: 0 0 1.15rem 0;
    }
    .ka-side-h {
      font-size: 0.64rem;
      font-weight: 600;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--ka-muted);
      margin: 0.35rem 0 0.42rem 0;
    }
    .ka-side-divider {
      height: 1px;
      background: var(--ka-line);
      margin: 0.15rem 0 0.65rem 0;
      opacity: 0.95;
    }
    .ka-side-sp { height: 0.55rem; }
    p.ka-lib-file {
      margin: 0.12rem 0 !important;
      font-size: 0.835rem !important;
      line-height: 1.38 !important;
      color: var(--ka-text);
    }
    .ka-lib-status {
      color: var(--ka-muted);
      font-size: 0.72rem;
      font-weight: 500;
    }
    .ka-lib-empty {
      font-size: 0.79rem !important;
      color: var(--ka-muted) !important;
      margin: 0 0 0.25rem 0 !important;
      line-height: 1.48;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
      padding-bottom: 0.3rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label p {
      font-size: 0.76rem !important;
      font-weight: 600 !important;
      color: var(--ka-text) !important;
      letter-spacing: 0.01em;
    }
    section[data-testid="stFileUploaderDropzone"] {
      min-height: 2.85rem;
      padding: 0.55rem 0.65rem;
      border-radius: 0.65rem !important;
      border-style: dashed !important;
      border-width: 1.5px !important;
      border-color: var(--ka-line) !important;
      background: var(--ka-surface) !important;
      transition: border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
    }
    section[data-testid="stFileUploaderDropzone"]:hover {
      border-color: rgba(37, 99, 235, 0.28) !important;
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.06);
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
      border-radius: 0.55rem !important;
      border-color: var(--ka-line) !important;
      background: var(--ka-surface) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
      border-radius: 0.65rem !important;
      font-weight: 600 !important;
      letter-spacing: 0.01em;
      box-shadow: 0 1px 3px rgba(37, 99, 235, 0.22);
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
      border-radius: 0.65rem !important;
      font-weight: 500 !important;
    }
    [data-testid="stSidebar"] table {
      width: 100%;
      font-size: 0.75rem;
      line-height: 1.42;
      border-collapse: collapse;
      margin: 0.3rem 0 0 0;
    }
    [data-testid="stSidebar"] table th,
    [data-testid="stSidebar"] table td {
      border-bottom: 1px solid var(--ka-line);
      padding: 0.42rem 0.35rem 0.42rem 0;
      vertical-align: top;
      text-align: left;
    }
    [data-testid="stSidebar"] table th { font-weight: 600; color: var(--ka-text); }
    [data-testid="stSidebar"] table td { color: var(--ka-muted); }
    .ka-hero-wrap {
      text-align: center;
      padding: 2.75rem 1.5rem 1.5rem;
      max-width: 34rem;
      margin: 0 auto 0.75rem auto;
      border-radius: 1.15rem;
      border: 1px solid var(--ka-line);
      background: var(--ka-surface);
      box-shadow: 0 4px 24px rgba(15, 23, 42, 0.055);
    }
    .ka-hero-wrap .ka-hero-title {
      font-weight: 600;
      letter-spacing: -0.038em;
      margin: 0 0 0.75rem 0;
      font-size: clamp(1.42rem, 3.6vw, 1.82rem);
      line-height: 1.18;
      color: var(--ka-text);
    }
    .ka-hero-wrap .ka-value {
      margin: 0 0 1.1rem 0;
      font-size: 0.98rem;
      line-height: 1.58;
      font-weight: 400;
      color: var(--ka-muted);
    }
    .ka-best-for {
      margin: 0 auto;
      max-width: 28rem;
      font-size: 0.84rem;
      line-height: 1.58;
      color: var(--ka-muted);
      text-align: center;
    }
    p.ka-empty-label {
      text-align: center;
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-weight: 600;
      color: var(--ka-muted);
      margin: 1.75rem 0 0.85rem 0 !important;
    }
    .main .stButton > button[kind="secondary"] {
      border-radius: 0.7rem !important;
      font-size: 0.88rem !important;
      padding: 0.5rem 0.75rem !important;
      transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .ka-empty-pad { height: 1.65rem; }
    /* Streamlit “Dark” theme (Settings); mirrors prefers-color-scheme dark tokens */
    .stApp[data-theme="dark"] {
      --ka-text: rgba(248, 250, 252, 0.94);
      --ka-muted: rgba(248, 250, 252, 0.52);
      --ka-line: rgba(255, 255, 255, 0.1);
      --ka-sidebar: rgba(15, 23, 42, 0.98);
      --ka-surface: rgba(30, 41, 59, 0.55);
      --ka-chat-user: rgba(30, 41, 59, 0.75);
      --ka-chat-asst: rgba(15, 23, 42, 0.55);
      --ka-accent: rgba(96, 165, 250, 0.45);
      --ka-app-bg-top: #0f172a;
      --ka-app-bg-mid: #111827;
      --ka-app-bg-bot: #0b1220;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.pop("_ka_sync_toast", False):
    st.toast("Synced", icon=st_icons.TOAST_SYNC)
if st.session_state.pop("_ka_ingest_toast", False):
    st.toast("Added to library", icon=st_icons.TOAST_LIBRARY)

with st.sidebar:
    st.markdown(f'<p class="ka-brand-title">{html.escape(APP_NAME)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="ka-brand-sub">{html.escape(SIDEBAR_CAPTION)}</p>', unsafe_allow_html=True)
    render_sidebar_chats()
    render_sidebar_documents_and_actions(raw_dir, faiss_folder)

cs, co, tk = read_settings()
task_mode, summarize_scope = read_task_settings()

pending = st.session_state.pop("pending_starter", None)
if pending:
    sid = st.session_state.active_chat_id
    chat_store.append_message(sid, "user", pending, {})
    maybe_set_session_title_from_first_user_message(sid, pending)
    st.session_state["_pending_assistant"] = {
        "q": pending,
        "ingest_warn": None,
        "library_refreshed": False,
        "cs": cs,
        "co": co,
        "tk": tk,
        "task_mode": task_mode,
        "summarize_scope": summarize_scope,
        "turn_source": "starter",
    }
    st.rerun()

if not st.session_state.messages:
    render_empty_state_hero()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg)
        else:
            render_user_message(msg.get("content") or "")

run_pending_assistant_turn(
    raw_dir,
    faiss_folder,
    cs=cs,
    co=co,
    tk=tk,
    task_mode=task_mode,
    summarize_scope=summarize_scope,
)

render_composer_task_hint(task_mode, summarize_scope)
prompt = st.chat_input("Ask anything…")
if prompt:
    q = prompt.strip()
    if q:
        if debug_service.debug_enabled():
            debug_service.merge(task_mode=task_mode, summarize_scope=summarize_scope)
        sid = st.session_state.active_chat_id
        chat_store.append_message(sid, "user", q, {})
        maybe_set_session_title_from_first_user_message(sid, q)
        ingest_warn: str | None = None
        library_refreshed = False
        with st.spinner("Saving uploads…"):
            try:
                ingest_warn, library_refreshed = ingest_composer_attachments(
                    raw_dir,
                    faiss_folder,
                    chunk_size=cs,
                    chunk_overlap=co,
                )
            except Exception as exc:
                debug_service.debug_begin_turn("chat_input")
                debug_service.merge(exception_summary=debug_service.short_exc(exc))
                ingest_warn = MSG_UPLOAD_FAILED
                library_refreshed = False
        st.session_state["_pending_assistant"] = {
            "q": q,
            "ingest_warn": ingest_warn,
            "library_refreshed": library_refreshed,
            "cs": cs,
            "co": co,
            "tk": tk,
            "task_mode": task_mode,
            "summarize_scope": summarize_scope,
            "turn_source": "chat_input",
        }
        if library_refreshed:
            st.session_state["_ka_ingest_toast"] = True
        st.rerun()
