"""Load optional local env files without overriding process/host variables.

Secret resolution order (each variable):

1. **Process environment** — always wins (Render, Docker, CI, shell). Never overwritten
   by files.
2. **Files (merged):** values from ``.env`` and ``.env.local`` are combined; if both
   define the same key, **``.env.local`` wins** (preferred for local secrets).
3. **Application:** for keys other than ``OPENAI_API_KEY``, a variable is set from files
   only if it is **missing** from the process environment.
4. **``OPENAI_API_KEY``:** a blank or whitespace-only process value is treated as unset so
   ``.env`` / ``.env.local`` can still supply the key.

**Never loaded:** ``.env.example``, ``.env.local.template``, or any other tracked sample
file — only ``.env`` and ``.env.local`` at the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def is_openai_key_placeholder(key: str) -> tuple[bool, str]:
    """
    True if ``key`` is an obvious sample / template value (eval preflight, smoke checks).

    Real keys are typically ``sk-…`` and do not contain template phrases below.
    """
    lower = (key or "").strip().lower()
    if not lower:
        return True, "empty"
    if "your_openai_api_key_here" in lower:
        return True, "template your_openai_api_key_here"
    if "sk-your-openai-api-key-here" in lower or "your-openai-api-key-here" in lower:
        return True, "legacy .env.example template"
    if lower.startswith("sk-your") or lower.startswith("sk-your-"):
        return True, "example prefix sk-your*"
    if "your-api-key-here" in lower:
        return True, "substring your-api-key-here"
    if lower in ("placeholder", "sk-placeholder", "sk-placeholderkey"):
        return True, "literal placeholder"
    return False, ""


def _env_file_nonempty(key: str) -> bool:
    return bool((os.environ.get(key) or "").strip())


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
        if key == "OPENAI_API_KEY":
            if _env_file_nonempty(key):
                continue
            os.environ[key] = value
            continue
        if key not in os.environ:
            os.environ[key] = value


def describe_openai_key_for_diagnostics(root: Path | None = None) -> dict[str, Any]:
    """
    Safe metadata for troubleshooting eval / API startup (no secret material).

    Call after :func:`load_repo_dotenv` if you need post-load effective key stats.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = None  # type: ignore[assignment]

    base = (root or _REPO_ROOT).resolve()
    p_env = base / ".env"
    p_local = base / ".env.local"
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()

    file_local = file_root = None
    if dotenv_values:
        if p_local.is_file():
            file_local = (dotenv_values(p_local).get("OPENAI_API_KEY") or "").strip() or None
        if p_env.is_file():
            file_root = (dotenv_values(p_env).get("OPENAI_API_KEY") or "").strip() or None

    masked = "(empty)"
    if len(key) >= 8:
        masked = f"{key[:4]}…{key[-3:]} (len={len(key)})"
    elif key:
        masked = f"(short, len={len(key)})"

    ph, why = is_openai_key_placeholder(key)
    placeholder_hits: list[str] = [] if not ph else [why]

    would_resolve_from = "none"
    if key:
        if file_local and key == file_local:
            would_resolve_from = ".env.local"
        elif file_root and key == file_root:
            would_resolve_from = ".env"
        elif file_local or file_root:
            would_resolve_from = "process_environment_or_other"
        else:
            would_resolve_from = "process_environment"

    return {
        "repo_root": str(base),
        "dotenv_files": {
            ".env": {"exists": p_env.is_file(), "has_openai_key": bool(file_root)},
            ".env.local": {"exists": p_local.is_file(), "has_openai_key": bool(file_local)},
        },
        "effective_key_present": bool(key),
        "effective_key_masked": masked,
        "effective_key_length": len(key),
        "placeholder_heuristic_hits": placeholder_hits,
        "is_placeholder_template": ph,
        "inferred_value_source": would_resolve_from,
    }
