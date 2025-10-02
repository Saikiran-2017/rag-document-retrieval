"""Lightweight latency capture merged into debug panel and optional structured logs."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

from app.services import debug_service

logger = logging.getLogger(__name__)


def perf_log_enabled() -> bool:
    return os.environ.get("KA_PERF_LOG", "1").strip().lower() not in ("0", "false", "no")


@contextmanager
def timed_phase(name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = round((time.perf_counter() - t0) * 1000, 2)
        if debug_service.debug_enabled():
            debug_service.merge(**{f"latency_ms_{name}": ms})


def record_phase_ms(name: str, ms: float) -> None:
    if debug_service.debug_enabled():
        debug_service.merge(**{f"latency_ms_{name}": round(ms, 2)})


def log_answer_pipeline_metrics(
    *,
    routing: str,
    ttft_ms: float | None = None,
    total_ms: float | None = None,
    retrieval_ms: float | None = None,
    generation_ms: float | None = None,
    rewrite_ms: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Structured performance line for dashboards / grep (enable with ``KA_PERF_LOG=1``)."""
    if not perf_log_enabled():
        return
    parts = [
        f"routing={routing!r}",
        f"ttft_ms={ttft_ms}",
        f"total_ms={total_ms}",
        f"retrieval_ms={retrieval_ms}",
        f"generation_ms={generation_ms}",
        f"rewrite_ms={rewrite_ms}",
    ]
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v!r}")
    logger.info("ka_perf answer_pipeline %s", " ".join(parts))
    if debug_service.debug_enabled():
        m: dict[str, Any] = {
            "perf_routing": routing,
            "latency_ms_ttft": ttft_ms,
            "latency_ms_total": total_ms,
            "latency_ms_retrieval": retrieval_ms,
            "latency_ms_generation": generation_ms,
            "latency_ms_rewrite": rewrite_ms,
        }
        if extra:
            m.update(extra)
        debug_service.merge(**{k: v for k, v in m.items() if v is not None})
