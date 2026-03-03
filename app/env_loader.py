"""Load optional local env files without overriding process/host variables.

Priority for each variable:
1. Already set in ``os.environ`` (Render, Docker, CI, shell) — never overwritten.
2. ``.env`` (gitignored) — defaults for local dev.
3. ``.env.local`` (gitignored) — overrides ``.env`` for the same key when the host
   did not already set it (preferred place for local secrets).

Tracked files like ``.env.example`` are never loaded here.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_repo_dotenv(root: Path | None = None) -> None:
    """Merge ``.env`` then ``.env.local`` into the environment (host wins)."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    base = (root or _REPO_ROOT).resolve()
    merged = {**dotenv_values(base / ".env"), **dotenv_values(base / ".env.local")}
    for key, value in merged.items():
        if value is None:
            continue
        if key not in os.environ:
            os.environ[key] = value
