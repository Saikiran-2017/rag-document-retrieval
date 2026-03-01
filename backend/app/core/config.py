"""API configuration: paths resolve to the same ``data/`` layout as Streamlit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.ingestion.loader import get_default_raw_dir
from app.retrieval.vector_store import get_default_faiss_folder


@dataclass(frozen=True)
class Settings:
    raw_dir: Path
    faiss_dir: Path
    default_chunk_size: int
    default_chunk_overlap: int
    default_top_k: int


def get_cors_allow_origins() -> list[str]:
    """
    Origins allowed for browser clients (Next.js dev server, etc.).

    ``KA_CORS_ORIGINS`` is a comma-separated list. If unset, defaults cover
    local Next.js on common hosts. Do not use ``*`` with ``allow_credentials=True``.
    """
    raw = os.environ.get("KA_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",
    ]


@lru_cache
def get_settings() -> Settings:
    raw = Path(os.environ.get("KA_RAW_DIR", str(get_default_raw_dir()))).resolve()
    fss = Path(os.environ.get("KA_FAISS_DIR", str(get_default_faiss_folder()))).resolve()
    cs = int(os.environ.get("KA_DEFAULT_CHUNK_SIZE", "500"))
    co = int(os.environ.get("KA_DEFAULT_CHUNK_OVERLAP", "80"))
    tk = int(os.environ.get("KA_DEFAULT_TOP_K", "3"))
    return Settings(
        raw_dir=raw,
        faiss_dir=fss,
        default_chunk_size=max(100, min(4000, cs)),
        default_chunk_overlap=max(0, min(2000, co)),
        default_top_k=max(1, min(12, tk)),
    )
