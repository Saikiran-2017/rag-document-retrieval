"""
Knowledge Assistant — Streamlit UI. Backend modules unchanged.

Run from project root: streamlit run streamlit_app.py
"""

from __future__ import annotations

import html
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from app.ingestion.loader import get_default_raw_dir, load_raw_directory
from app.llm.generator import GroundedAnswer, generate_grounded_answer
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
TAGLINE = "Ask questions grounded in the documents you add here."
SUPPORTED_EXT = {".pdf", ".docx", ".txt"}
PREVIEW_CHARS = 300

STARTER_QUESTIONS = [
    "What are the main themes in these documents?",
    "Summarize the key points I should know.",
    "What definitions or terms are explained?",
]

MSG_NO_DOCS = "Upload documents in the sidebar, then choose Sync Documents, before asking questions."

WHY_THIS_APP_MD = """
Tools like ChatGPT, Claude, and Microsoft Copilot are built for breadth: general knowledge and many kinds of tasks. **This assistant answers from the files you upload and sync**—not from the open web or the model's training data alone.

Replies stay scoped to **your library**, with **citations** and optional **excerpts** so you can see what text informed each answer. That fits workflows where the source of truth is your own material: review, study, policies, contracts, or internal docs.

It is a narrower tool by design, not a substitute for a general assistant in every situation.

| Capability | General assistants | This app |
|:-----------|:-------------------|:---------|
| Your uploaded files | Depends on product | Yes, after sync |
| Citations in answers | Varies | Built in |
| Search scope | Broad | Your synced library |
| See supporting text | Often limited | Excerpts available |
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
        return False, "Overlap must be smaller than segment size (Settings).", 0

    docs = load_raw_directory(raw_dir)
    if not docs:
        return False, "Add at least one document, then sync again.", 0

    chunks = chunk_ingested_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        return False, "No readable text found. Check that files are not empty.", 0

    embeddings = create_openai_embeddings(model=DEFAULT_EMBEDDING_MODEL)
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


def _answer_question(
    question: str,
    raw_dir: Path,
    faiss_folder: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> tuple[GroundedAnswer | None, list[RetrievedChunk] | None, str | None]:
    ok, sync_msg = _ensure_index_matches_library(
        raw_dir,
        faiss_folder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not ok:
        return None, None, sync_msg

    embeddings = create_openai_embeddings(model=DEFAULT_EMBEDDING_MODEL)
    store = load_faiss_index(
        folder_path=faiss_folder,
        index_name=DEFAULT_INDEX_NAME,
        embeddings=embeddings,
    )
    nvec = faiss_vector_count(store)
    if nvec == 0:
        return None, None, MSG_NO_DOCS

    k = min(int(top_k), nvec)
    hits = retrieve_top_k(store, question, k=k)
    result = generate_grounded_answer(question, hits)
    return result, hits, None


def _append_assistant_turn(
    res: GroundedAnswer | None,
    hits: list[RetrievedChunk] | None,
    err: str | None,
) -> None:
    if err and not res:
        st.session_state.messages.append({"role": "assistant", "content": err})
    elif res:
        st.session_state.messages.append(_assistant_payload(res, hits))


def _read_settings() -> tuple[int, int, int]:
    cs = int(st.session_state.get("settings_chunk_size", st.session_state.get("adv_chunk_size", 500)))
    co = int(st.session_state.get("settings_chunk_overlap", st.session_state.get("adv_chunk_overlap", 80)))
    tk = int(st.session_state.get("settings_top_k", st.session_state.get("adv_top_k", 4)))
    return cs, co, tk


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_sync_fingerprint" not in st.session_state:
        st.session_state.kb_sync_fingerprint = None
    if "settings_chunk_size" not in st.session_state and "adv_chunk_size" in st.session_state:
        st.session_state.settings_chunk_size = st.session_state["adv_chunk_size"]
    if "settings_chunk_overlap" not in st.session_state and "adv_chunk_overlap" in st.session_state:
        st.session_state.settings_chunk_overlap = st.session_state["adv_chunk_overlap"]
    if "settings_top_k" not in st.session_state and "adv_top_k" in st.session_state:
        st.session_state.settings_top_k = st.session_state["adv_top_k"]


def _new_chat() -> None:
    st.session_state.messages = []


def _display_path_hint(path_str: str | None) -> str | None:
    if not path_str:
        return None
    try:
        p = Path(path_str)
        return p.name if p.name else path_str
    except (TypeError, ValueError):
        return path_str


def _render_assistant_message(msg: dict[str, Any]) -> None:
    st.markdown(msg.get("content") or "")
    sources = msg.get("sources") or []
    legacy_passages = msg.get("retrieved_passages") or []
    excerpts = msg.get("excerpts") or legacy_passages

    if sources:
        with st.expander("Sources", expanded=False):
            for i, s in enumerate(sources):
                if i:
                    st.markdown('<div class="ka-src-gap"></div>', unsafe_allow_html=True)
                sn = s.get("source_number", "?")
                st.markdown(
                    f"**[{sn}]** {s.get('source_name', '')} &middot; p. {s.get('page_label', 'n/a')}"
                )
                cid = s.get("chunk_id")
                if cid:
                    st.caption(cid)
                fp = _display_path_hint(s.get("file_path"))
                if fp:
                    st.caption(fp)

    if excerpts:
        with st.expander("Supporting excerpts", expanded=False):
            for i, p in enumerate(excerpts):
                if i:
                    st.markdown('<div class="ka-ex-gap"></div>', unsafe_allow_html=True)
                fn = p.get("file_name", "")
                cid = p.get("chunk_id", "")
                pg = p.get("page_label", "-")
                st.markdown(f"**{fn}** &middot; `{cid}` &middot; Page {pg}")
                st.caption(p.get("preview_text") or "")


_init_session()

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
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
    [data-testid="stChatMessage"] { padding-top: 0.35rem !important; padding-bottom: 0.65rem !important; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
      line-height: 1.62; font-size: 0.98rem; margin-bottom: 0.45em;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }
    [data-testid="stExpander"] { margin-top: 0.35rem !important; margin-bottom: 0.15rem !important; }
    [data-testid="stExpander"] details { border: none !important; }
    [data-testid="stExpander"] summary { font-weight: 500; letter-spacing: 0.01em; padding: 0.35rem 0; }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0.35rem; font-size: 0.92rem; }
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] { margin-top: 0.15rem; opacity: 0.92; }
    .ka-src-gap { height: 0.65rem; }
    .ka-ex-gap { height: 0.75rem; border-top: 1px solid rgba(128,128,128,0.14); margin: 0.5rem 0 0.65rem 0; }
    [data-testid="stSidebarContent"] { padding-top: 1rem !important; padding-bottom: 1.25rem !important; }
    [data-testid="stSidebarContent"] .stMarkdown h3 { margin-bottom: 0.2rem; font-weight: 600; letter-spacing: -0.02em; }
    [data-testid="stSidebarContent"] [data-testid="stCaption"] { margin-top: 0.15rem; margin-bottom: 0.85rem; line-height: 1.45; }
    .ka-side-h { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.55;
      margin: 0.9rem 0 0.45rem 0; font-weight: 500; }
    .ka-side-sp { height: 0.5rem; }
    .ka-hero-wrap { text-align: center; padding: 2.25rem 1rem 2rem; max-width: 34rem; margin: 0 auto; }
    .ka-hero-wrap h1 { font-weight: 600; letter-spacing: -0.035em; margin: 0 0 0.55rem 0;
      font-size: clamp(1.45rem, 4vw, 1.8rem); line-height: 1.2; border: none; }
    .ka-hero-wrap .ka-tag { opacity: 0.88; font-size: 1.02rem; line-height: 1.55; margin: 0 0 1.35rem 0; }
    .ka-hero-wrap .ka-hint { opacity: 0.68; font-size: 0.88rem; line-height: 1.5; margin: 0 0 1.6rem 0; }
    .ka-hero-wrap .ka-lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em; opacity: 0.5;
      margin: 0 0 0.75rem 0; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption(TAGLINE)

    st.markdown('<div class="ka-side-sp"></div>', unsafe_allow_html=True)

    st.file_uploader(
        "Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="doc_upload",
        help="PDF, Word, or plain text",
    )
    uploaded = st.session_state.get("doc_upload")

    st.markdown('<p class="ka-side-h">Settings</p>', unsafe_allow_html=True)
    st.session_state.settings_chunk_size = st.number_input(
        "Segment size (characters)",
        100,
        4000,
        int(st.session_state.get("settings_chunk_size", 500)),
        step=50,
        help="Larger segments carry more context per piece of text.",
    )
    st.session_state.settings_chunk_overlap = st.number_input(
        "Overlap (characters)",
        0,
        2000,
        int(st.session_state.get("settings_chunk_overlap", 80)),
        step=10,
    )
    st.session_state.settings_top_k = st.number_input(
        "Sources per answer",
        1,
        20,
        int(st.session_state.get("settings_top_k", 4)),
        step=1,
    )

    st.markdown('<div class="ka-side-sp"></div>', unsafe_allow_html=True)

    if st.button("Sync Documents", type="primary", use_container_width=True):
        _save_uploads_to_raw(uploaded, raw_dir)
        cs, co, _ = _read_settings()
        with st.spinner("Syncing..."):
            ok, err_msg, _ = _rebuild_knowledge_index(
                raw_dir,
                faiss_folder,
                chunk_size=cs,
                chunk_overlap=co,
            )
        if ok:
            st.session_state.kb_sync_fingerprint = _library_fingerprint(raw_dir)
            st.success("Library updated.")
        else:
            st.warning(err_msg)

    files_now = _list_raw_files(raw_dir)
    if files_now:
        st.markdown('<p class="ka-side-h">Library</p>', unsafe_allow_html=True)
        for p in files_now:
            st.text(p.name)
    else:
        st.caption("No files yet.")

    st.divider()

    if st.button("New Chat", use_container_width=True):
        _new_chat()
        st.rerun()

    st.divider()
    with st.expander("Why this app", expanded=False):
        st.markdown(WHY_THIS_APP_MD)

cs, co, tk = _read_settings()

pending = st.session_state.pop("pending_starter", None)
if pending:
    st.session_state.messages.append({"role": "user", "content": pending})
    err: str | None = None
    res: GroundedAnswer | None = None
    hits: list[RetrievedChunk] | None = None
    with st.spinner("Working..."):
        try:
            res, hits, err = _answer_question(
                pending,
                raw_dir,
                faiss_folder,
                chunk_size=cs,
                chunk_overlap=co,
                top_k=tk,
            )
        except Exception as e:
            err = str(e)
            res, hits = None, None
    _append_assistant_turn(res, hits, err)

if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="ka-hero-wrap">
          <h1>{html.escape(APP_NAME)}</h1>
          <p class="ka-tag">{html.escape(TAGLINE)}</p>
          <p class="ka-hint">Add files in the sidebar, sync, then type below or pick an example.</p>
          <p class="ka-lbl">Try asking</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    for i, (col, q) in enumerate(zip((c1, c2, c3), STARTER_QUESTIONS)):
        with col:
            if st.button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_starter = q
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_message(msg)
        else:
            st.markdown(msg["content"])

prompt = st.chat_input("Ask about your documents")
if prompt:
    q = prompt.strip()
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        err2: str | None = None
        res2: GroundedAnswer | None = None
        hits2: list[RetrievedChunk] | None = None
        with st.spinner("Working..."):
            try:
                res2, hits2, err2 = _answer_question(
                    q,
                    raw_dir,
                    faiss_folder,
                    chunk_size=cs,
                    chunk_overlap=co,
                    top_k=tk,
                )
            except Exception as e:
                err2 = str(e)
                res2, hits2 = None, None
        _append_assistant_turn(res2, hits2, err2)
        st.rerun()
