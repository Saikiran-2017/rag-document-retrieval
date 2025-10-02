"""
Structured reliability logging for answer turns (routing, weak retrieval hints, validation cues).

Enable with ``KA_RELIABILITY_LOG=1`` (or ``true`` / ``yes``). Uses the ``rag.reliability`` logger;
configure handlers in your deployment if you want JSON lines files.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("rag.reliability")

WEAK_RETRIEVAL_ROUTINGS = frozenset(
    {
        "general_weak_no_web",
        "web_weak_docs",
        "general_web_thin",
        "general_empty_index",
        "general_retrieval_failed",
    }
)


def reliability_log_enabled() -> bool:
    return os.environ.get("KA_RELIABILITY_LOG", "").strip().lower() in ("1", "true", "yes")


def infer_validation_concern(turn: Any) -> str | None:
    """Detect user-visible notes that imply citation or grounding fixes."""
    n = (turn.assistant_note or "").lower()
    if "withheld" in n:
        return "source_citation_withheld"
    if "removed from the answer" in n or "did not match retrieved" in n:
        return "web_link_sanitized"
    if "verify in sources" in n or "retrieved excerpts" in n:
        return "lexical_support_warning"
    return None


def infer_reliability_flags(
    routing: str,
    turn: Any,
    *,
    dbg: dict[str, Any],
) -> dict[str, Any]:
    """Derive boolean flags for dashboards / alerting (no LLM calls)."""
    r = routing or ""
    weak_path = r in WEAK_RETRIEVAL_ROUTINGS or r == "general_sync_fallback"
    retrieval_ran = bool(dbg.get("retrieval_ran"))
    hits = int(dbg.get("retrieval_hit_count") or 0)
    return {
        "weak_retrieval_or_fallback_route": weak_path,
        "retrieval_ran_no_hits": retrieval_ran and hits == 0 and r not in ("error_empty", "general_fastpath"),
        "validation_concern": infer_validation_concern(turn),
        "fallback_to_general": bool(dbg.get("fallback_to_general")),
    }


def log_reliability_turn(turn: Any, **dbg: Any) -> None:
    """
    Emit one JSON line per turn when enabled. Includes routing and derived flags.

    For **incorrect routing** vs product expectations, use ``eval/phase15_scenarios.json``
    and manual or scripted checks; this logger records actuals for diffing.
    """
    if not reliability_log_enabled():
        return
    routing = str(dbg.get("routing", "") or "")
    flags = infer_reliability_flags(routing, turn, dbg=dbg)
    payload: dict[str, Any] = {
        "event": "reliability_turn",
        "routing": routing,
        "turn_mode": getattr(turn, "mode", None),
        "retrieval_ran": dbg.get("retrieval_ran"),
        "retrieval_hit_count": dbg.get("retrieval_hit_count"),
        "fallback_to_general": dbg.get("fallback_to_general"),
        "no_retrieval_fastpath": dbg.get("no_retrieval_fastpath"),
        "exception_summary": (dbg.get("exception_summary") or "")[:300] or None,
        **flags,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    try:
        line = json.dumps(payload, default=str)[:8192]
    except (TypeError, ValueError):
        line = json.dumps({"event": "reliability_turn", "routing": routing, "error": "serialize_failed"})
    logger.info("%s", line)


def log_validation_failure(kind: str, **fields: Any) -> None:
    """Log grounding / citation fixes (enable with ``KA_RELIABILITY_LOG``)."""
    if not reliability_log_enabled():
        return
    payload = {"event": "validation_failure", "kind": kind, **fields}
    try:
        logger.info("%s", json.dumps(payload, default=str)[:4096])
    except (TypeError, ValueError):
        logger.info('{"event":"validation_failure","kind":"serialize_error"}')
