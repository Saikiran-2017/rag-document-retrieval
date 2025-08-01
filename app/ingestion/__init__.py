"""Document ingestion: load PDF, DOCX, and TXT into structured records."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "IngestedDocument",
    "get_default_raw_dir",
    "load_file",
    "load_raw_directory",
    "print_ingestion_summary",
]

if TYPE_CHECKING:
    from app.ingestion.loader import (
        IngestedDocument,
        get_default_raw_dir,
        load_file,
        load_raw_directory,
        print_ingestion_summary,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = import_module("app.ingestion.loader")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
