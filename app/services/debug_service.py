"""Developer debug flag, per-turn state, and optional sidebar debug panel."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

# Set True (or env KA_DEBUG=1) to show a compact sidebar debug panel.
# Does not expose API keys; avoids full stack traces in the UI.
DEVELOPER_DEBUG_MODE = False


def debug_enabled() -> bool:
    return DEVELOPER_DEBUG_MODE or os.environ.get("KA_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _dev_debug_bucket() -> dict[str, Any] | None:
    """Streamlit session dict for debug merges, or None when not in a Streamlit script (e.g. FastAPI)."""
    try:
        return st.session_state.setdefault("_dev_debug", {})
    except Exception:
        return None


def debug_begin_turn(source: str) -> None:
    if not debug_enabled():
        return
    try:
        st.session_state["_dev_debug"] = {"turn_source": source}
    except Exception:
        pass


def merge(**kwargs: Any) -> None:
    if not debug_enabled():
        return
    d = _dev_debug_bucket()
    if d is None:
        return
    for k, v in kwargs.items():
        if v is not None:
            d[k] = v


def short_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc!s}"[:400]


def render_debug_panel() -> None:
    if not debug_enabled():
        return
    with st.expander("Developer debug", expanded=False):
        d = dict(st.session_state.get("_dev_debug", {}))
        d["composer_reset"] = int(st.session_state.get("composer_reset", 0))
        d["message_count"] = len(st.session_state.get("messages") or [])
        st.json(d)
