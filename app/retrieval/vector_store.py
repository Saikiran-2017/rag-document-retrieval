"""
Build OpenAI embeddings, store them in a local FAISS index, and retrieve top-k chunks by query.

Indexing (embed + save) and retrieval (query embedding + similarity search) share the same
``OpenAIEmbeddings`` model so query vectors match stored vectors. Answer generation is a
separate phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import get_openai_api_key
from app.utils.chunker import TextChunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_INDEX_NAME = "rag_index"

# FAISS default in LangChain uses L2 distance; lower values mean closer / more similar.
ScoreKind = Literal["faiss_l2"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_default_faiss_folder() -> Path:
    """Directory under ``data/indexes/`` where the FAISS files are stored."""
    return _project_root() / "data" / "indexes" / "faiss_store"


def create_openai_embeddings(
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> OpenAIEmbeddings:
    """
    LangChain wrapper around OpenAI embeddings.

    Uses the same API key as the rest of the app via :func:`app.config.get_openai_api_key`.
    """
    return OpenAIEmbeddings(model=model, api_key=get_openai_api_key())


def _metadata_for_storage(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    FAISS + pickle tolerate most Python types; None can confuse some downstream JSON paths.

    Empty string keeps keys stable for UI and filtering later.
    """
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        out[key] = "" if value is None else value
    return out


def chunks_to_documents(chunks: list[TextChunk]) -> list[Document]:
    """Map each :class:`TextChunk` to a LangChain :class:`Document` (content + metadata + id)."""
    documents: list[Document] = []
    for chunk in chunks:
        documents.append(
            Document(
                id=chunk.chunk_id,
                page_content=chunk.text,
                metadata=_metadata_for_storage(chunk.metadata),
            )
        )
    return documents


def build_faiss_from_chunks(
    chunks: list[TextChunk],
    *,
    embeddings: OpenAIEmbeddings | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """
    Embed all chunk texts and build an in-memory FAISS index with metadata in the docstore.

    Parameters
    ----------
    chunks
        Output from :func:`app.utils.chunker.chunk_ingested_documents` (or helpers).
    embeddings
        Optional pre-built embeddings client; if omitted, one is created with ``embedding_model``.
    embedding_model
        OpenAI embedding model name (must match when loading an index from disk later).
    """
    if not chunks:
        raise ValueError("No chunks to index. Ingest and chunk documents first.")

    embedder = embeddings or create_openai_embeddings(model=embedding_model)
    documents = chunks_to_documents(chunks)
    logger.info("Embedding %s chunk(s) into FAISS...", len(documents))
    return FAISS.from_documents(documents, embedder)


def save_faiss_index(
    store: FAISS,
    folder_path: str | Path | None = None,
    *,
    index_name: str = DEFAULT_INDEX_NAME,
) -> Path:
    """
    Persist FAISS binary index plus pickled docstore (vectors + texts + metadata).

    Creates ``{index_name}.faiss`` and ``{index_name}.pkl`` under ``folder_path``.
    """
    path = Path(folder_path).resolve() if folder_path is not None else get_default_faiss_folder()
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path), index_name=index_name)
    logger.info("Saved FAISS index to %s (%s.*)", path, index_name)
    return path


def load_faiss_index(
    folder_path: str | Path | None = None,
    *,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embeddings: OpenAIEmbeddings | None = None,
) -> FAISS:
    """
    Load a FAISS index from disk.

    The embedding model (and API key) must match what was used when the index was built,
    so vector dimensionality stays consistent with the stored matrix.
    """
    path = Path(folder_path).resolve() if folder_path is not None else get_default_faiss_folder()
    embedder = embeddings or create_openai_embeddings(model=embedding_model)
    # Local pickle from your own ``data/indexes`` tree; required flag is LangChain's safety default.
    return FAISS.load_local(
        str(path),
        embedder,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def faiss_index_files_exist(
    folder_path: str | Path | None = None,
    *,
    index_name: str = DEFAULT_INDEX_NAME,
) -> bool:
    """Return True if both ``{index_name}.faiss`` and ``{index_name}.pkl`` exist."""
    folder = Path(folder_path).resolve() if folder_path is not None else get_default_faiss_folder()
    return (folder / f"{index_name}.faiss").is_file() and (folder / f"{index_name}.pkl").is_file()


def faiss_vector_count(store: FAISS) -> int:
    """Number of vectors in the FAISS index (equals chunk count if built from chunks)."""
    return int(store.index.ntotal)


def print_index_build_summary(
    store: FAISS,
    *,
    folder: Path,
    index_name: str,
    embedding_model: str,
    phase: str,
) -> None:
    """Short console summary for demos and sanity checks."""
    n = faiss_vector_count(store)
    print(f"[{phase}] embedding_model={embedding_model!r}")
    print(f"[{phase}] vectors in index: {n}")
    print(f"[{phase}] on-disk folder: {folder}")
    print(f"[{phase}] files: {index_name}.faiss, {index_name}.pkl")


@dataclass(frozen=True)
class RetrievedChunk:
    """
    One search hit, ready for answer generation (prompt context + citations).

    ``distance`` is FAISS L2 distance (LangChain default): **smaller = closer** to the query
    in embedding space. It is not a bounded "probability."
    """

    rank: int
    page_content: str
    metadata: dict[str, Any]
    distance: float
    score_kind: ScoreKind = "faiss_l2"

    def as_langchain_document(self) -> Document:
        """Rebuild a LangChain :class:`Document` for RAG prompt construction."""
        cid = self.metadata.get("chunk_id")
        doc_id = str(cid) if cid not in (None, "") else None
        return Document(
            id=doc_id,
            page_content=self.page_content,
            metadata=dict(self.metadata),
        )


def retrieve_top_k(
    store: FAISS,
    query: str,
    *,
    k: int = 4,
) -> list[RetrievedChunk]:
    """
    Embed ``query`` with the store's embedding model and return the top-``k`` chunks.

    Uses ``similarity_search_with_score`` so each hit includes an L2 **distance** (lower is better).

    Parameters
    ----------
    store
        Loaded or freshly built ``FAISS`` instance (must use the same embeddings as at index time).
    query
        Natural-language question or keywords.
    k
        Number of documents to return (clamped to index size by FAISS).
    """
    if not query.strip():
        raise ValueError("Query must be non-empty.")

    pairs = store.similarity_search_with_score(query, k=k)
    results: list[RetrievedChunk] = []
    for rank, (doc, distance) in enumerate(pairs):
        results.append(
            RetrievedChunk(
                rank=rank,
                page_content=doc.page_content,
                metadata=dict(doc.metadata),
                distance=float(distance),
            )
        )
    return results


def print_retrieval_report(hits: list[RetrievedChunk], *, query: str) -> None:
    """Pretty-print retrieval results for terminal demos (ASCII-friendly)."""
    print(f"Query: {query!r}")
    print(f"Hits: {len(hits)} (score_kind=faiss_l2, lower distance = more similar)\n")
    for h in hits:
        print(f"--- rank {h.rank}  distance={h.distance:.6f} ---")
        print(f"  chunk_id: {h.metadata.get('chunk_id', '')!r}")
        print(f"  document_id: {h.metadata.get('document_id', '')!r}")
        print(f"  source_name: {h.metadata.get('source_name', '')!r}")
        print(f"  page_number: {h.metadata.get('page_number', '')!r}")
        preview = h.page_content[:200] + ("..." if len(h.page_content) > 200 else "")
        print(f"  text preview: {preview!r}")
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from app.ingestion.loader import load_raw_directory
    from app.utils.chunker import chunk_ingested_documents

    folder = get_default_faiss_folder()
    index_name = DEFAULT_INDEX_NAME
    model = DEFAULT_EMBEDDING_MODEL

    print("=== Phase 5 demo: ingest -> chunk -> FAISS (load or build) -> retrieve ===\n")

    raw_docs = load_raw_directory()
    chunks = chunk_ingested_documents(raw_docs, chunk_size=500, chunk_overlap=80)
    if not chunks:
        print("No chunks produced. Add text/PDF/DOCX under data/raw/ and run again.")
        raise SystemExit(1)

    print(f"Chunks from ingestion pipeline: {len(chunks)}\n")

    embeddings = create_openai_embeddings(model=model)

    if faiss_index_files_exist(folder, index_name=index_name):
        print("Found existing FAISS files; loading index...\n")
        store = load_faiss_index(
            folder_path=folder,
            index_name=index_name,
            embeddings=embeddings,
        )
    else:
        print("No FAISS index on disk; building from current chunks and saving...\n")
        store = build_faiss_from_chunks(chunks, embeddings=embeddings)
        save_faiss_index(store, folder_path=folder, index_name=index_name)

    print_index_build_summary(
        store, folder=folder, index_name=index_name, embedding_model=model, phase="index ready"
    )

    sample_query = "What is chunking and metadata used for in this RAG sample?"
    print()
    hits = retrieve_top_k(store, sample_query, k=min(4, faiss_vector_count(store)))
    print_retrieval_report(hits, query=sample_query)

    if hits:
        print("OK: RetrievedChunk.as_langchain_document() is ready for the LLM prompt step.")
    else:
        print("No hits returned (unexpected if index is non-empty).")

# Improved 2024-08-05

# Improved 2024-09-02

# Improved 2024-09-24

# Improved 2024-10-17

# Improved 2024-08-05

# Improved 2024-09-02

# Improved 2025-08-05

# Improved 2025-09-02

# Improved 2025-09-24

# Improved 2025-10-17
