"""
RAG Document Retrieval — Streamlit UI.

Run from project root:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

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

SUPPORTED_EXT = {".pdf", ".docx", ".txt"}


def _list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        return []
    return sorted(
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def _clear_qa_state() -> None:
    for key in ("rag_question", "rag_result", "rag_hits"):
        st.session_state.pop(key, None)


st.set_page_config(
    page_title="RAG-Based Document Retrieval",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RAG-Based Document Retrieval System")
st.caption(
    "Upload documents, build a local vector index, then ask questions. "
    "Answers are grounded in retrieved passages with source citations."
)

raw_dir = get_default_raw_dir()
raw_dir.mkdir(parents=True, exist_ok=True)
faiss_folder = get_default_faiss_folder()

with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (characters)", min_value=100, max_value=4000, value=500, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=80, step=10)
    top_k = st.number_input("Top K retrieval", min_value=1, max_value=20, value=4, step=1)
    st.divider()
    st.caption(f"Embedding model: `{DEFAULT_EMBEDDING_MODEL}`")
    index_ok = faiss_index_files_exist(faiss_folder, index_name=DEFAULT_INDEX_NAME)
    st.caption("Index status: **ready**" if index_ok else "**not built**")

# --- 1. Upload ---
st.subheader("1. Documents")
st.write("Upload PDF, DOCX, or TXT files, then save them into `data/raw/`.")

uploaded = st.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Save uploaded files to data/raw", type="primary"):
    if not uploaded:
        st.warning("No files selected. Choose files above, then click Save.")
    else:
        n = 0
        for f in uploaded:
            dest = raw_dir / f.name
            dest.write_bytes(f.getvalue())
            n += 1
        st.success(f"Saved {n} file(s) to `{raw_dir}`. Rebuild the index to include new text.")
        _clear_qa_state()

existing = _list_raw_files(raw_dir)
if existing:
    st.caption(f"Files currently in library: {len(existing)} — " + ", ".join(p.name for p in existing[:8]) + (" ..." if len(existing) > 8 else ""))
else:
    st.info("No supported files in `data/raw/` yet. Upload and save, or add files manually.")

# --- 2. Index ---
st.subheader("2. Vector index")
st.write("Ingest all files from `data/raw/`, chunk text, embed with OpenAI, and save a FAISS index locally.")

if st.button("Build / Rebuild Index"):
    if chunk_overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size. Adjust sidebar settings.")
    else:
        try:
            docs = load_raw_directory(raw_dir)
            if not docs:
                st.warning("No documents to index. Upload and save files first (or place PDF/DOCX/TXT in data/raw/).")
            else:
                with st.spinner("Building index (embeddings API calls)..."):
                    chunks = chunk_ingested_documents(
                        docs,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                    if not chunks:
                        st.warning("Ingestion produced no text chunks. Check that files are not empty.")
                    else:
                        embeddings = create_openai_embeddings(model=DEFAULT_EMBEDDING_MODEL)
                        store = build_faiss_from_chunks(chunks, embeddings=embeddings)
                        save_faiss_index(store, folder_path=faiss_folder, index_name=DEFAULT_INDEX_NAME)
                        _clear_qa_state()
                        st.success(
                            f"Index built: **{faiss_vector_count(store)}** vectors from **{len(chunks)}** chunk(s). "
                            f"Saved under `{faiss_folder}`."
                        )
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Index build failed: {e}")

# --- 3. Question ---
st.subheader("3. Ask a question")
question = st.text_input("Your question", placeholder="e.g. What does the policy say about ...?", key="question_input")

if st.button("Ask Question"):
    q = (question or "").strip()
    if not q:
        st.warning("Enter a question before clicking Ask.")
    elif not faiss_index_files_exist(faiss_folder, index_name=DEFAULT_INDEX_NAME):
        st.warning("Build the index first (step 2).")
    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                embeddings = create_openai_embeddings(model=DEFAULT_EMBEDDING_MODEL)
                store = load_faiss_index(
                    folder_path=faiss_folder,
                    index_name=DEFAULT_INDEX_NAME,
                    embeddings=embeddings,
                )
                nvec = faiss_vector_count(store)
                if nvec == 0:
                    st.error("Index is empty. Rebuild the index.")
                else:
                    k = min(int(top_k), nvec)
                    hits: list[RetrievedChunk] = retrieve_top_k(store, q, k=k)
                    result: GroundedAnswer = generate_grounded_answer(q, hits)
                    st.session_state["rag_question"] = q
                    st.session_state["rag_result"] = result
                    st.session_state["rag_hits"] = hits
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Request failed: {e}")

# --- Display last answer (persists across reruns until cleared) ---
if st.session_state.get("rag_result") is not None:
    res: GroundedAnswer = st.session_state["rag_result"]
    q_done = st.session_state.get("rag_question", "")
    hits_done: list[RetrievedChunk] = st.session_state.get("rag_hits") or []

    st.divider()
    st.markdown(f"**Question:** {q_done!r}")
    st.markdown("### Answer")
    st.markdown(res.answer)

    st.markdown("### Sources")
    if not res.sources:
        st.caption("No sources in context (empty retrieval or index issue).")
    else:
        for src in res.sources:
            with st.expander(f"[SOURCE {src.source_number}] — {src.source_name or '(unknown file)'}", expanded=False):
                st.write(f"**chunk_id:** `{src.chunk_id}`")
                st.write(f"**page:** {src.page_label}")
                if src.file_path:
                    st.caption(f"path: `{src.file_path}`")

        st.markdown("#### Retrieved chunk previews (FAISS L2 distance — lower is closer)")
        if not hits_done:
            st.caption("No retrieval rows stored for this run.")
        else:
            for h in hits_done:
                label = h.metadata.get("chunk_id", f"rank_{h.rank}")
                prev = h.page_content[:400] + ("..." if len(h.page_content) > 400 else "")
                st.markdown(f"**{label}** · distance `{h.distance:.4f}`")
                st.text(prev)

# Enhanced 2024-08-15

# Enhanced 2024-09-09

# Enhanced 2024-10-03

# Enhanced 2024-10-23

# Enhanced 2024-08-15

# Enhanced 2024-09-09
