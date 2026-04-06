"""Developer debug flag, per-turn state, and optional sidebar debug panel."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import streamlit as st

_retrieval_logger = logging.getLogger("rag.retrieval")

# Set True (or env KA_DEBUG=1) to show a compact sidebar debug panel.
# Does not expose API keys; avoids full stack traces in the UI.
DEVELOPER_DEBUG_MODE = False


def debug_enabled() -> bool:
    return DEVELOPER_DEBUG_MODE or os.environ.get("KA_DEBUG", "").strip().lower() in ("1", "true", "yes")


def retrieval_debug_enabled() -> bool:
    """Structured retrieval pipeline traces (stderr + logger); works outside Streamlit."""
    v = os.environ.get("KA_RETRIEVAL_DEBUG", "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    return debug_enabled()


def log_retrieval_event(event: str, **data: Any) -> None:
    """
    One JSON line per event to stderr when KA_RETRIEVAL_DEBUG=1 (or KA_DEBUG=1).

    Safe for logs: no API keys; chunk previews are truncated.
    """
    if not retrieval_debug_enabled():
        return
    payload: dict[str, Any] = {"event": event}
    for k, v in data.items():
        try:
            json.dumps(v, default=str)
            payload[k] = v
        except TypeError:
            payload[k] = str(v)[:500]
    line = json.dumps(payload, default=str, ensure_ascii=False)
    print(line, file=sys.stderr, flush=True)
    _retrieval_logger.info("%s", line)


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
