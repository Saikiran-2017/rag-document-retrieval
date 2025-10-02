"""Load and validate Phase 15 scenario definitions (JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_scenarios_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "eval" / "phase15_scenarios.json"


def load_scenarios(path: Path | None = None) -> dict[str, Any]:
    p = path or default_scenarios_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "scenarios" not in data:
        raise ValueError("scenarios file must be a JSON object with key 'scenarios'")
    return data


def validate_scenarios(data: dict[str, Any]) -> list[str]:
    """Return human-readable errors; empty list means OK."""
    errors: list[str] = []
    seen: set[str] = set()
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list):
        return ["'scenarios' must be a list"]
    for i, s in enumerate(scenarios):
        if not isinstance(s, dict):
            errors.append(f"scenario[{i}] must be an object")
            continue
        sid = s.get("id")
        if not sid or not isinstance(sid, str):
            errors.append(f"scenario[{i}] missing string id")
        elif sid in seen:
            errors.append(f"duplicate scenario id: {sid}")
        else:
            seen.add(sid)
        for key in ("name", "preconditions", "user_message", "expected_routing", "expected_mode"):
            if key not in s:
                errors.append(f"{sid or i}: missing '{key}'")
        er = s.get("expected_routing")
        if er is not None and not isinstance(er, str) and not isinstance(er, list):
            errors.append(f"{sid or i}: expected_routing must be string or list of strings")
        em = s.get("expected_mode")
        if em is not None and not isinstance(em, str) and not isinstance(em, list):
            errors.append(f"{sid or i}: expected_mode must be string or list of strings")
    return errors


def routing_matches(actual_routing: str, expected: str | list[str] | None) -> bool:
    if expected is None:
        return True
    if isinstance(expected, str):
        return actual_routing == expected
    return actual_routing in expected


def mode_matches(actual_mode: str, expected: str | list[str] | None) -> bool:
    if expected is None:
        return True
    if isinstance(expected, str):
        return actual_mode == expected
    return actual_mode in expected
