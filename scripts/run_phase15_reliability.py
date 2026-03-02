#!/usr/bin/env python3
"""
Phase 15: end-to-end reliability validation entrypoint.

1. Validates ``eval/phase15_scenarios.json`` structure (15+ scenarios).
2. Runs pytest routing + chat-store tests (no API keys required).

Usage (from repo root)::

    python scripts/run_phase15_reliability.py

Optional: set ``KA_RELIABILITY_LOG=1`` while using the Streamlit app to emit JSON
``reliability_turn`` and ``validation_failure`` lines on the ``rag.reliability`` logger.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from app.reliability.scenarios import default_scenarios_path, load_scenarios, validate_scenarios

    data = load_scenarios(default_scenarios_path())
    errs = validate_scenarios(data)
    if errs:
        print("phase15_scenarios.json validation FAILED:")
        for e in errs:
            print(" -", e)
        return 1
    n = len(data["scenarios"])
    print(f"phase15_scenarios.json OK ({n} scenarios).")
    if n < 15:
        print("WARNING: fewer than 15 scenarios.")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT / "tests" / "test_reliability_routing.py"),
        str(ROOT / "tests" / "test_chat_store_reliability.py"),
        "-v",
        "--tb=short",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
