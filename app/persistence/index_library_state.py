"""On-disk state for incremental FAISS builds (file content hashes + chunk settings)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_STATE_NAME = "index_library_state.json"


def state_path(faiss_folder: Path) -> Path:
    return Path(faiss_folder).resolve() / _STATE_NAME


def load_state(faiss_folder: Path) -> dict[str, Any] | None:
    p = state_path(faiss_folder)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_state(faiss_folder: Path, data: dict[str, Any]) -> None:
    p = state_path(faiss_folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
