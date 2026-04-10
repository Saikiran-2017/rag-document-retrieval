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
    # Local dev + E2E: include the configured web port when present (e.g. Playwright on 3100).
    extra_ports: set[int] = set()
    for key in ("E2E_WEB_PORT", "KA_WEB_PORT", "PORT_WEB"):
        v = (os.environ.get(key) or "").strip()
        if v.isdigit():
            extra_ports.add(int(v))
    # Default Next dev port.
    extra_ports.add(3000)
    # Common alternate dev ports.
    extra_ports.add(3100)
    extra_ports.add(3200)
    origins: list[str] = []
    for p in sorted(extra_ports):
        origins.extend(
            [
                f"http://localhost:{p}",
                f"http://127.0.0.1:{p}",
                f"http://[::1]:{p}",
            ]
        )
    return origins


@lru_cache
def get_settings() -> Settings:
    raw = Path(os.environ.get("KA_RAW_DIR", str(get_default_raw_dir()))).resolve()
    fss = Path(os.environ.get("KA_FAISS_DIR", str(get_default_faiss_folder()))).resolve()
    cs = int(os.environ.get("KA_DEFAULT_CHUNK_SIZE", "900"))
    co = int(os.environ.get("KA_DEFAULT_CHUNK_OVERLAP", "120"))
    tk = int(os.environ.get("KA_DEFAULT_TOP_K", "5"))
    return Settings(
        raw_dir=raw,
        faiss_dir=fss,
        default_chunk_size=max(100, min(4000, cs)),
        default_chunk_overlap=max(0, min(2000, co)),
        default_top_k=max(1, min(12, tk)),
    )
