"""Shared utilities (e.g. chunking)."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "TextChunk",
    "chunk_ingested_documents",
    "chunk_raw_directory",
    "chunk_single_file",
    "print_chunk_summary",
]

if TYPE_CHECKING:
    from app.utils.chunker import (
        TextChunk,
        chunk_ingested_documents,
        chunk_raw_directory,
        chunk_single_file,
        print_chunk_summary,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = import_module("app.utils.chunker")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
