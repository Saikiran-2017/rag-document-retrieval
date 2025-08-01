"""Retrieval package: vector store build/load and top-k similarity search."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_INDEX_NAME",
    "RetrievedChunk",
    "build_faiss_from_chunks",
    "chunks_to_documents",
    "create_openai_embeddings",
    "faiss_index_files_exist",
    "faiss_vector_count",
    "get_default_faiss_folder",
    "load_faiss_index",
    "print_index_build_summary",
    "print_retrieval_report",
    "retrieve_top_k",
    "save_faiss_index",
]

if TYPE_CHECKING:
    from app.retrieval.vector_store import (
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_INDEX_NAME,
        RetrievedChunk,
        build_faiss_from_chunks,
        chunks_to_documents,
        create_openai_embeddings,
        faiss_index_files_exist,
        faiss_vector_count,
        get_default_faiss_folder,
        load_faiss_index,
        print_index_build_summary,
        print_retrieval_report,
        retrieve_top_k,
        save_faiss_index,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = import_module("app.retrieval.vector_store")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
