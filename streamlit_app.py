"""Streamlit UI for Knowledge Assistant. Run: streamlit run streamlit_app.py"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import streamlit as st

from app.ingestion.loader import get_default_raw_dir, load_raw_directory
from app.llm.generator import (
    GroundedAnswer,
    generate_general_answer,
    generate_grounded_answer,
    retrieval_is_useful,
)
from app.retrieval.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    RetrievedChunk,
    build_faiss_from_chunks,
    create_openai_embeddings,
    faiss_index_files_exist,
    faiss_vector_count,
    get_default_faiss_folder,
    load_faiss_index,
    retrieve_top_k,
    save_faiss_index,
)
from app.utils.chunker import chunk_ingested_documents

APP_NAME = "Knowledge Assistant"
# First-screen value prop (also used where TAGLINE was referenced).
EMPTY_STATE_VALUE_PROP = "Ask anything. Add files when you want answers from your documents."
TAGLINE = EMPTY_STATE_VALUE_PROP
SIDEBAR_CAPTION = "Your questions, your documents."
SUPPORTED_EXT = {".pdf", ".docx", ".txt"}
PREVIEW_CHARS = 300
# Sidebar UI: max value for "Sources per answer" (retrieval top_k).
SIDEBAR_TOP_K_MAX = 12
# Default retrieval width (lower = faster retrieval + smaller prompts).
DEFAULT_TOP_K = 3

# If the query clearly does not ask about documents, skip FAISS load + retrieval (same UX).
_DOC_QUERY_HINT = re.compile(
    r"\b(document|documents|file|files|pdf|upload|uploaded|page|pages|passage|passages|"
    r"excerpt|excerpts|cite|citation|source\b|sources\b|section|library|chunk|"
    r"these files|my files|the text|summarize|summary|key points|themes|outline|"
    r"extract|according to|from the|in the document)\b",
    re.I,
)


def _wants_no_retrieval_fastpath(query: str) -> bool:
    q = query.strip()
    if len(q) < 4 or len(q) > 220:
        return False
    if _DOC_QUERY_HINT.search(q):
        return False
    return True


@st.cache_resource(show_spinner=False)
def _cached_openai_embeddings(model: str) -> Any:
    return create_openai_embeddings(model=model)


def _faiss_disk_version(folder: Path, index_name: str) -> str:
    faiss_f = folder / f"{index_name}.faiss"
    pkl_f = folder / f"{index_name}.pkl"
    if not faiss_f.is_file() or not pkl_f.is_file():
        return "missing"
    return f"{faiss_f.stat().st_mtime_ns}-{pkl_f.stat().st_mtime_ns}"


@st.cache_resource(show_spinner=False)
def _cached_faiss_store(faiss_folder_str: str, index_name: str, cache_key: str) -> Any:
    """Load FAISS from disk; cache invalidates when library fingerprint or on-disk index changes."""
    return load_faiss_index(
        folder_path=Path(faiss_folder_str),
        index_name=index_name,
        embeddings=_cached_openai_embeddings(DEFAULT_EMBEDDING_MODEL),
    )

STARTER_QUESTIONS = [
    "Give three concise tips for clearer writing.",
    "Summarize the key points I should know.",
    "What themes or terms show up in my documents?",
]

# User-facing copy only (no indexing / embedding / vector jargon).
MSG_LIBRARY_UPDATED = "Library updated."
MSG_PREPARE_DOCS_FAILED = "Couldn't prepare documents. Please try again."
MSG_PREPARE_SETTINGS_HINT = "Couldn't prepare documents. Adjust Settings, then try again."
MSG_READ_FILES_FAILED = "Couldn't read text from those files. Try another file or check they aren't empty."
MSG_NO_DOCS = "Add a file below your message, send again, then ask about it."
MSG_CHAT_UNEXPECTED = "Couldn't complete that. Please try again."
MSG_EMPTY_MESSAGE = "Please enter a message."


@dataclass
class AssistantTurn:
    """Result of routing: document-grounded vs general assistant vs error."""

    mode: Literal["grounded", "general", "error"]
    text: str
    grounded: GroundedAnswer | None = None
    hits: list[RetrievedChunk] | None = None
    error: str | None = None


ABOUT_APP_MD = """
Mainstream assistants are built for **breadth**: many topics and many kinds of tasks.

This app is aimed at **your uploaded documents**: add files, ask questions, and when your material is used you can see **sources** and supporting excerpts. You can also ask **general questions** without uploads; those answers use the model's general knowledge, not your library.

It is meant for that workflow when you want answers tied to your own files, not for every situation, and not a replacement for every product.
"""


def _list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        return []
    return sorted(
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def _library_fingerprint(raw_dir: Path) -> tuple[tuple[str, int], ...]:
    files = _list_raw_files(raw_dir)
    return tuple((p.name, int(p.stat().st_mtime_ns)) for p in files)


def _save_uploads_to_raw(uploaded_files: list[Any] | None, raw_dir: Path) -> int:
    if not uploaded_files:
        return 0
    n = 0
    for f in uploaded_files:
        (raw_dir / f.name).write_bytes(f.getvalue())
        n += 1
    return n


def _preview_text(text: str, max_len: int = PREVIEW_CHARS) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "..."


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
                "preview_text": _preview_text(h.page_content),
            }
        )
    return out


def _assistant_payload(res: GroundedAnswer, hits: list[RetrievedChunk] | None) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": res.answer,
        "grounded": True,
        "sources": [asdict(s) for s in res.sources],
        "excerpts": _hits_to_excerpts(hits),
    }


def _rebuild_knowledge_index(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[bool, str, int]:
    if chunk_overlap >= chunk_size:
        return False, MSG_PREPARE_SETTINGS_HINT, 0

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

    embeddings = _cached_openai_embeddings(DEFAULT_EMBEDDING_MODEL)
    store = build_faiss_from_chunks(chunks, embeddings=embeddings)
    save_faiss_index(store, folder_path=faiss_folder, index_name=DEFAULT_INDEX_NAME)
    nvec = faiss_vector_count(store)
    return True, "", nvec


def _ensure_index_matches_library(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[bool, str]:
    fp = _library_fingerprint(raw_dir)
    if not fp:
        return False, MSG_NO_DOCS

    synced = st.session_state.get("kb_sync_fingerprint")
    index_ready = faiss_index_files_exist(faiss_folder, index_name=DEFAULT_INDEX_NAME)

    if index_ready and synced == fp:
        return True, ""

    ok, msg, _ = _rebuild_knowledge_index(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if ok:
        st.session_state.kb_sync_fingerprint = fp
        return True, ""
    return False, msg


def answer_user_query(
    query: str,
    *,
    raw_dir: Path,
    faiss_folder: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> AssistantTurn:
    """
    Route to a general assistant reply when there is no library or retrieval is weak;
    otherwise return a document-grounded answer with sources.
    """
    if not query.strip():
        return AssistantTurn(mode="error", text=MSG_EMPTY_MESSAGE, error=MSG_EMPTY_MESSAGE)

    if not _library_fingerprint(raw_dir):
        try:
            text = generate_general_answer(query)
        except Exception:
            return AssistantTurn(mode="error", text=MSG_CHAT_UNEXPECTED, error=MSG_CHAT_UNEXPECTED)
        return AssistantTurn(mode="general", text=text)

    ok, sync_msg = _ensure_index_matches_library(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not ok:
        return AssistantTurn(mode="error", text=sync_msg, error=sync_msg)

    fp_now = _library_fingerprint(raw_dir)
    if _wants_no_retrieval_fastpath(query):
        try:
            text = generate_general_answer(query)
        except Exception:
            return AssistantTurn(mode="error", text=MSG_CHAT_UNEXPECTED, error=MSG_CHAT_UNEXPECTED)
        return AssistantTurn(mode="general", text=text)

    cache_key = f"{fp_now!r}|{_faiss_disk_version(faiss_folder, DEFAULT_INDEX_NAME)}"
    store = _cached_faiss_store(str(faiss_folder.resolve()), DEFAULT_INDEX_NAME, cache_key)
    nvec = faiss_vector_count(store)
    if nvec == 0:
        try:
            text = generate_general_answer(query)
        except Exception:
            return AssistantTurn(mode="error", text=MSG_CHAT_UNEXPECTED, error=MSG_CHAT_UNEXPECTED)
        return AssistantTurn(mode="general", text=text)

    k = min(int(top_k), nvec)
    hits = retrieve_top_k(store, query, k=k)
    if retrieval_is_useful(hits):
        try:
            ga = generate_grounded_answer(query, hits)
        except Exception:
            return AssistantTurn(mode="error", text=MSG_CHAT_UNEXPECTED, error=MSG_CHAT_UNEXPECTED)
        return AssistantTurn(mode="grounded", text=ga.answer, grounded=ga, hits=hits)

    try:
        text = generate_general_answer(query)
    except Exception:
        return AssistantTurn(mode="error", text=MSG_CHAT_UNEXPECTED, error=MSG_CHAT_UNEXPECTED)
    return AssistantTurn(mode="general", text=text)


def _with_status_note(msg: dict[str, Any], note: str | None) -> dict[str, Any]:
    if note and note.strip():
        return {**msg, "status_note": note.strip()}
    return msg


def _append_assistant_turn(
    turn: AssistantTurn,
    *,
    ingest_note: str | None = None,
) -> None:
    note = ingest_note.strip() if ingest_note else None
    if turn.mode == "error":
        st.session_state.messages.append(
            _with_status_note({"role": "assistant", "content": turn.error or turn.text}, note)
        )
    elif turn.mode == "general":
        st.session_state.messages.append(_with_status_note({"role": "assistant", "content": turn.text}, note))
    elif turn.mode == "grounded" and turn.grounded:
        st.session_state.messages.append(_with_status_note(_assistant_payload(turn.grounded, turn.hits), note))


def _read_settings() -> tuple[int, int, int]:
    cs = int(st.session_state.get("settings_chunk_size", st.session_state.get("adv_chunk_size", 500)))
    co = int(st.session_state.get("settings_chunk_overlap", st.session_state.get("adv_chunk_overlap", 80)))
    tk = int(st.session_state.get("settings_top_k", st.session_state.get("adv_top_k", DEFAULT_TOP_K)))
    tk = max(1, min(SIDEBAR_TOP_K_MAX, tk))
    return cs, co, tk


def _init_session() -> None:
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


def _new_chat() -> None:
    """Clear chat only; library files and index on disk are unchanged."""
    st.session_state.messages = []


def _display_path_hint(path_str: str | None) -> str | None:
    if not path_str:
        return None
    try:
        p = Path(path_str)
        return p.name if p.name else path_str
    except (TypeError, ValueError):
        return path_str


def _is_grounded_assistant_message(msg: dict[str, Any]) -> bool:
    """Document-grounded turns only (show Sources / excerpts expanders)."""
    if msg.get("grounded"):
        return True
    if msg.get("sources") or msg.get("excerpts") or msg.get("retrieved_passages"):
        return True
    return False


def _render_user_message(content: str) -> None:
    st.markdown(content)


def _render_sources_expander(sources: list[dict[str, Any]]) -> None:
    with st.expander("Sources", expanded=False):
        for i, s in enumerate(sources):
            if i:
                st.markdown('<div class="ka-src-gap"></div>', unsafe_allow_html=True)
            sn = s.get("source_number", "?")
            st.markdown(
                f"**[{sn}]** {s.get('source_name', '')} &middot; p. {s.get('page_label', 'n/a')}"
            )
            fp = _display_path_hint(s.get("file_path"))
            if fp:
                st.caption(fp)


def _render_excerpts_expander(excerpts: list[dict[str, Any]]) -> None:
    with st.expander("Supporting excerpts", expanded=False):
        for i, p in enumerate(excerpts):
            if i:
                st.markdown('<div class="ka-ex-gap"></div>', unsafe_allow_html=True)
            fn = p.get("file_name", "")
            pg = p.get("page_label", "-")
            st.markdown(f"**{html.escape(str(fn))}** &middot; Page {html.escape(str(pg))}")
            st.caption(p.get("preview_text") or "")


def _render_grounded_expanders(msg: dict[str, Any]) -> None:
    sources = msg.get("sources") or []
    legacy_passages = msg.get("retrieved_passages") or []
    excerpts = msg.get("excerpts") or legacy_passages
    if sources:
        _render_sources_expander(sources)
    if excerpts:
        _render_excerpts_expander(excerpts)


def _render_assistant_message(msg: dict[str, Any]) -> None:
    # Answer first; optional status (e.g. library update) after the body, muted.
    st.markdown(msg.get("content") or "")
    if msg.get("status_note"):
        st.markdown(f'<p class="ka-msg-status">{html.escape(str(msg["status_note"]))}</p>', unsafe_allow_html=True)
    if _is_grounded_assistant_message(msg):
        _render_grounded_expanders(msg)


def _ingest_composer_attachments(
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str | None, bool]:
    """
    Save files from the current composer widget and rebuild the library index.
    Returns (error_message_or_none, did_save_any_file).
    """
    rid = int(st.session_state.get("composer_reset", 0))
    key = f"composer_{rid}"
    uploaded = st.session_state.get(key)
    n = _save_uploads_to_raw(uploaded, raw_dir)
    if n == 0:
        return None, False
    ok, err_msg, _ = _rebuild_knowledge_index(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if ok:
        st.session_state.kb_sync_fingerprint = _library_fingerprint(raw_dir)
        st.session_state.composer_reset = rid + 1
        return None, True
    return err_msg, True


_init_session()

st.set_page_config(
    page_title=APP_NAME,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

raw_dir = get_default_raw_dir()
raw_dir.mkdir(parents=True, exist_ok=True)
faiss_folder = get_default_faiss_folder()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem !important; padding-bottom: 1.5rem !important;
      max-width: 42rem !important; margin-left: auto !important; margin-right: auto !important; }
    div[data-testid="stChatInput"] { padding-bottom: 0.5rem; padding-top: 0.35rem; }
    [data-testid="stChatMessage"] {
      padding-top: 0.25rem !important;
      padding-bottom: 0.25rem !important;
      margin-bottom: 1rem !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
      line-height: 1.65; font-size: 0.98rem; margin-bottom: 0.5em;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
      line-height: 1.5; font-size: 0.9rem; border-radius: 0.35rem;
    }
    p.ka-msg-status {
      margin: 0.65rem 0 0 0 !important;
      font-size: 0.78rem !important;
      line-height: 1.45 !important;
      opacity: 0.58 !important;
      letter-spacing: 0.01em;
    }
    [data-testid="stExpander"] { margin-top: 0.5rem !important; margin-bottom: 0.2rem !important; }
    [data-testid="stExpander"] details { border: none !important; background: transparent !important; }
    [data-testid="stExpander"] summary {
      font-weight: 500; font-size: 0.88rem; letter-spacing: 0.01em; padding: 0.4rem 0;
      color: inherit; opacity: 0.88;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0.35rem; font-size: 0.9rem; }
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] { margin-top: 0.12rem; opacity: 0.88; }
    .ka-src-gap { height: 0.55rem; }
    .ka-ex-gap { height: 0.6rem; margin: 0.45rem 0; border-top: 1px solid rgba(128,128,128,0.1); }
    [data-testid="stSidebar"] { background: rgba(128,128,128,0.045) !important; border-right: 1px solid rgba(128,128,128,0.12) !important; }
    [data-testid="stSidebarContent"] {
      padding-top: 0.75rem !important; padding-bottom: 1rem !important;
      font-size: 0.92rem !important;
    }
    [data-testid="stSidebarContent"] .stMarkdown h3 {
      margin-bottom: 0.1rem !important; font-weight: 600; font-size: 1.05rem !important;
      letter-spacing: -0.02em; color: rgba(0,0,0,0.82);
    }
    [data-testid="stSidebarContent"] [data-testid="stCaption"] {
      margin-top: 0.05rem !important; margin-bottom: 0.6rem !important; line-height: 1.4;
      font-size: 0.82rem !important; opacity: 0.72;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] { margin-top: 0.35rem !important; margin-bottom: 0.15rem !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary { font-size: 0.82rem !important; opacity: 0.85; }
    .ka-side-h {
      font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
      opacity: 0.45; margin: 0 0 0.35rem 0;
    }
    .ka-side-sp { height: 0.4rem; }
    p.ka-lib-file { margin: 0.12rem 0 !important; font-size: 0.86rem !important; line-height: 1.35 !important; opacity: 0.9; }
    .ka-lib-empty { font-size: 0.8rem !important; opacity: 0.55; margin: 0 0 0.25rem 0 !important; }
    .ka-hero-wrap {
      text-align: center; padding: 2.75rem 1.25rem 0.5rem; max-width: 36rem; margin: 0 auto;
    }
    .ka-hero-wrap h1 {
      font-weight: 600; letter-spacing: -0.04em; margin: 0 0 0.75rem 0; border: none;
      font-size: clamp(1.55rem, 4.2vw, 1.95rem); line-height: 1.18; color: rgba(0,0,0,0.9);
    }
    .ka-hero-wrap .ka-value {
      margin: 0; font-size: 1.04rem; line-height: 1.55; font-weight: 400;
      color: rgba(0,0,0,0.62); max-width: 28rem; margin-left: auto; margin-right: auto;
    }
    p.ka-empty-label {
      text-align: center; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.11em;
      font-weight: 500; color: rgba(0,0,0,0.38); margin: 2rem 0 0.9rem 0 !important;
    }
    .ka-empty-pad { height: 1.25rem; }
    div[data-testid="stVerticalBlock"] > div[data-testid="stFileUploader"] {
      padding-bottom: 0.35rem; margin-bottom: 0; max-width: 42rem; margin-left: auto; margin-right: auto;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stFileUploader"] label p {
      font-size: 0.82rem !important; opacity: 0.72 !important; font-weight: 500 !important;
    }
    section[data-testid="stFileUploaderDropzone"] { min-height: 2.25rem; padding: 0.35rem 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption(SIDEBAR_CAPTION)

    st.markdown('<p class="ka-side-h">Library</p>', unsafe_allow_html=True)
    files_now = _list_raw_files(raw_dir)
    if files_now:
        for p in files_now:
            st.markdown(f'<p class="ka-lib-file">{html.escape(p.name)}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="ka-lib-empty">No files yet. Add them below the chat.</p>', unsafe_allow_html=True)

    st.markdown('<div class="ka-side-sp"></div>', unsafe_allow_html=True)

    if st.button("New chat", use_container_width=True, type="primary"):
        _new_chat()
        st.rerun()

    st.markdown('<div class="ka-side-sp"></div>', unsafe_allow_html=True)

    with st.expander("More options", expanded=False):
        _tk = max(1, min(SIDEBAR_TOP_K_MAX, int(st.session_state.get("settings_top_k", DEFAULT_TOP_K))))
        st.session_state.settings_top_k = st.select_slider(
            "Sources per answer",
            options=list(range(1, SIDEBAR_TOP_K_MAX + 1)),
            value=_tk,
            help="How many places in your files can inform one reply.",
        )
        st.session_state.settings_chunk_size = st.number_input(
            "Context length",
            100,
            4000,
            int(st.session_state.get("settings_chunk_size", 500)),
            step=50,
            help="More text per segment. Change only if you need different behavior.",
        )
        st.session_state.settings_chunk_overlap = st.number_input(
            "Overlap",
            0,
            2000,
            int(st.session_state.get("settings_chunk_overlap", 80)),
            step=10,
            help="Shared text between segments. Usually leave as-is.",
        )

    with st.expander("About", expanded=False):
        st.markdown(ABOUT_APP_MD)

cs, co, tk = _read_settings()

pending = st.session_state.pop("pending_starter", None)
if pending:
    st.session_state.messages.append({"role": "user", "content": pending})
    with st.spinner("Thinking..."):
        try:
            turn = answer_user_query(
                pending,
                raw_dir=raw_dir,
                faiss_folder=faiss_folder,
                chunk_size=cs,
                chunk_overlap=co,
                top_k=tk,
            )
        except Exception:
            turn = AssistantTurn(mode="error", text=MSG_CHAT_UNEXPECTED, error=MSG_CHAT_UNEXPECTED)
    _append_assistant_turn(turn)

if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="ka-hero-wrap">
          <h1>{html.escape(APP_NAME)}</h1>
          <p class="ka-value">{html.escape(EMPTY_STATE_VALUE_PROP)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="ka-empty-label">Suggestions</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (col, q) in enumerate(zip((c1, c2, c3), STARTER_QUESTIONS)):
        with col:
            if st.button(q, key=f"starter_{i}", use_container_width=True, type="secondary"):
                st.session_state.pending_starter = q
                st.rerun()
    st.markdown('<div class="ka-empty-pad"></div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_message(msg)
        else:
            _render_user_message(msg.get("content") or "")

rid = int(st.session_state.get("composer_reset", 0))
st.file_uploader(
    "Add files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key=f"composer_{rid}",
    label_visibility="visible",
    help="Used with your next message.",
)

prompt = st.chat_input("Ask anything")
if prompt:
    q = prompt.strip()
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        ingest_err: str | None = None
        turn2: AssistantTurn | None = None
        had_new_upload = False
        with st.spinner("Thinking..."):
            try:
                ingest_err, had_new_upload = _ingest_composer_attachments(
                    raw_dir,
                    faiss_folder,
                    chunk_size=cs,
                    chunk_overlap=co,
                )
                if ingest_err:
                    st.session_state.messages.append({"role": "assistant", "content": ingest_err})
                else:
                    turn2 = answer_user_query(
                        q,
                        raw_dir=raw_dir,
                        faiss_folder=faiss_folder,
                        chunk_size=cs,
                        chunk_overlap=co,
                        top_k=tk,
                    )
            except Exception:
                turn2 = AssistantTurn(mode="error", text=MSG_CHAT_UNEXPECTED, error=MSG_CHAT_UNEXPECTED)
        if not ingest_err and turn2 is not None:
            _append_assistant_turn(
                turn2,
                ingest_note=MSG_LIBRARY_UPDATED if had_new_upload else None,
            )
        st.rerun()
