#!/usr/bin/env python3
"""
Safe check: OPENAI_API_KEY is set, not a known template, with masked preview and source hint.

Does not print the full secret. Exit 0 = OK for local dev; 1 = missing/placeholder; 2 = bad usage.

Usage (repo root)::

    set PYTHONPATH=.
    .venv\\Scripts\\python.exe scripts/verify_openai_env.py

Optional live check (network)::

    .venv\\Scripts\\python.exe scripts/verify_openai_env.py --ping-openai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify OPENAI_API_KEY (safe, masked)")
    parser.add_argument(
        "--ping-openai",
        action="store_true",
        help="Call OpenAI models.list() (uses a tiny request; requires valid key).",
    )
    args = parser.parse_args()

    from app.env_loader import describe_openai_key_for_diagnostics, load_repo_dotenv

    load_repo_dotenv(ROOT)
    info = describe_openai_key_for_diagnostics(ROOT)
    not_placeholder = not info.get("is_placeholder_template", True)
    info["placeholder_detection_passed"] = not_placeholder
    info["ready_for_local_dev"] = bool(
        info.get("effective_key_present")
        and not_placeholder
        and int(info.get("effective_key_length") or 0) >= 24
    )

    # Do not add raw OPENAI_API_KEY to this dict
    print(json.dumps(info, indent=2, ensure_ascii=False))

    if not info.get("effective_key_present"):
        return 1
    if not not_placeholder:
        return 1
    if int(info.get("effective_key_length") or 0) < 24:
        return 1

    if args.ping_openai:
        import os

        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        try:
            from openai import OpenAI

            OpenAI(api_key=key).models.list()
            print(json.dumps({"openai_ping": "ok"}, indent=2))
        except Exception as exc:
            print(json.dumps({"openai_ping": "failed", "error": str(exc)[:200]}, indent=2))
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
