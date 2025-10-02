"""Server-Sent Events framing (one event per yield from the route generator)."""

from __future__ import annotations

import json
from typing import Any


def format_sse(data: dict[str, Any], *, event: str | None = None) -> str:
    """Return one SSE message block (ends with blank line). ``data`` is JSON-encoded."""
    payload = json.dumps(data, ensure_ascii=False)
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    # Multiline data fields are allowed in SSE; JSON is always single-line with dumps.
    lines.append(f"data: {payload}")
    lines.append("")
    return "\n".join(lines)
