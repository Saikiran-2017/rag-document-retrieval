"""Load optional local env files without overriding process/host variables.

**Non-secret keys:** merged from ``.env`` then ``.env.local`` (local wins on duplicates);
each is applied only if **missing** from the process environment (host wins).

**``OPENAI_API_KEY``** (snapshot process **once** at load start, then resolve):

1. If the snapshot is **non-empty and not a placeholder** → use it (CI/Docker/Render).
2. Else if ``.env.local`` has a **non-placeholder** value → use it.
3. Else if ``.env`` has a **non-placeholder** value → use it.
4. Else → remove ``OPENAI_API_KEY`` from the environment if it was only a placeholder
   (clear failure instead of silently keeping a sample key).

Placeholders are detected by :func:`is_openai_key_placeholder` (sample templates, not real keys).

**Never loaded:** ``.env.example``, ``.env.local.template``, or other tracked samples —
only ``.env`` and ``.env.local`` at the repo root. Windows paths use :class:`pathlib.Path`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Set by load_repo_dotenv for describe_openai_key_for_diagnostics / verify script.
_OPENAI_KEY_RESOLUTION_SOURCE: str = "none"


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


def _pick_openai_api_key(
    process_snapshot: str,
    local_val: str | None,
    root_val: str | None,
) -> tuple[str | None, str]:
    """First valid (non-placeholder) key wins: process, then ``.env.local``, then ``.env``."""
    proc = (process_snapshot or "").strip()
    if proc and not is_openai_key_placeholder(proc)[0]:
        return proc, "process_environment"
    for raw, label in ((local_val, ".env.local"), (root_val, ".env")):
        v = (raw or "").strip()
        if v and not is_openai_key_placeholder(v)[0]:
            return v, label
    return None, "none"


def load_repo_dotenv(root: Path | None = None) -> None:
    """Merge ``.env`` / ``.env.local`` into the environment; resolve ``OPENAI_API_KEY`` explicitly."""
    global _OPENAI_KEY_RESOLUTION_SOURCE
    base = (root or _REPO_ROOT).resolve()
    process_snapshot = (os.environ.get("OPENAI_API_KEY") or "").strip()
    try:
        from dotenv import dotenv_values
    except ImportError:
        _OPENAI_KEY_RESOLUTION_SOURCE = "none"
        return
    root_d = dotenv_values(base / ".env") or {}
    local_d = dotenv_values(base / ".env.local") or {}
    merged = {**root_d, **local_d}
    local_oai = local_d.get("OPENAI_API_KEY")
    root_oai = root_d.get("OPENAI_API_KEY")
    picked, src = _pick_openai_api_key(process_snapshot, local_oai, root_oai)
    _OPENAI_KEY_RESOLUTION_SOURCE = src
    if picked:
        os.environ["OPENAI_API_KEY"] = picked
    else:
        os.environ.pop("OPENAI_API_KEY", None)

    for key, value in merged.items():
        if value is None:
            continue
        if key == "OPENAI_API_KEY":
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

    would_resolve_from = _OPENAI_KEY_RESOLUTION_SOURCE
    if key and would_resolve_from == "none":
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
