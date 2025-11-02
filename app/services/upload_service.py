"""Save uploaded files to the raw library directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_EXT = {".pdf", ".docx", ".txt"}


def save_uploads_to_raw(uploaded_files: list[Any] | None, raw_dir: Path) -> tuple[int, list[str]]:
    """Write each uploaded file to raw_dir; returns (count saved, filenames)."""
    if not uploaded_files:
        return 0, []
    n = 0
    saved: list[str] = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in uploaded_files:
        dest = raw_dir / f.name
        dest.write_bytes(f.getvalue())
        saved.append(f.name)
        n += 1
    return n, saved
