#!/usr/bin/env python3
"""
Run the document QA gold eval (requires OPENAI_API_KEY and repo venv).

Secrets: use the process environment first (recommended for public clones), or a
gitignored ``.env.local`` / ``.env`` — never commit real keys.

Usage (repository root)::

    set OPENAI_API_KEY=...   # optional if .env.local already set
    set PYTHONPATH=.
    .venv\\Scripts\\python.exe scripts/run_document_qa_eval.py --json-report eval/_report_local.json

Requires ``rank-bm25`` and full ``requirements.txt`` in the same interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _preflight_openai() -> str | None:
    """Return error message if the key is missing or rejected; None if OK."""
    from app.env_loader import is_openai_key_placeholder

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return "OPENAI_API_KEY is not set."
    bad, why = is_openai_key_placeholder(key)
    if bad:
        return f"OPENAI_API_KEY appears to be a template placeholder ({why})."
    if len(key) < 24:
        return "OPENAI_API_KEY is too short to be valid."
    try:
        from openai import OpenAI

        OpenAI(api_key=key).models.list()
    except Exception as exc:
        return f"OpenAI API rejected the key or request failed: {exc}"
    return None


def _write_blocked_report(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "eval_status": "blocked",
                "reason": reason,
                "cases_total": 8,
                "aggregate": {"cases": 0, "passed": 0, "pass_rate": 0.0},
                "by_category": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run document QA eval harness")
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Path to gold_cases.json (default: eval/gold_cases.json)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Temp corpus/index directory (default: eval/_work)",
    )
    parser.add_argument("--json-report", type=Path, default=None, help="Write machine-readable summary")
    parser.add_argument(
        "--verbose-key",
        action="store_true",
        help="Print safe OPENAI_API_KEY diagnostics (masked); does not print the secret.",
    )
    args = parser.parse_args()

    from app.env_loader import describe_openai_key_for_diagnostics, load_repo_dotenv

    load_repo_dotenv(ROOT)

    if args.verbose_key or os.environ.get("KA_EVAL_KEY_DIAG", "").strip().lower() in ("1", "true", "yes"):
        diag = describe_openai_key_for_diagnostics(ROOT)
        print("=== OPENAI_API_KEY diagnostics (safe) ===", file=sys.stderr)
        print(json.dumps(diag, indent=2), file=sys.stderr)

    pre = _preflight_openai()
    if pre:
        print(f"ERROR: {pre}", file=sys.stderr)
        if not args.verbose_key and os.environ.get("KA_EVAL_KEY_DIAG", "").strip().lower() not in (
            "1",
            "true",
            "yes",
        ):
            snap = describe_openai_key_for_diagnostics(ROOT)
            print(
                "Safe key hint: "
                + json.dumps(
                    {
                        "effective_key_present": snap["effective_key_present"],
                        "effective_key_masked": snap["effective_key_masked"],
                        "placeholder_heuristic_hits": snap["placeholder_heuristic_hits"],
                        "dotenv_files": snap["dotenv_files"],
                        "inferred_value_source": snap["inferred_value_source"],
                    },
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
            print("Re-run with --verbose-key for full JSON.", file=sys.stderr)
        if args.json_report:
            _write_blocked_report(args.json_report, pre)
            print(f"Wrote blocked report to {args.json_report}", file=sys.stderr)
        return 2

    from eval.harness import format_report, run_eval
    from eval.scoring import aggregate_by_category, aggregate_rates

    rows = list(run_eval(cfg_path=args.gold, work_dir=args.work_dir))
    print(format_report(rows))

    agg = aggregate_rates([(a, d) for a, _, _, d in rows])
    by_cat = aggregate_by_category(rows)
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eval_status": "complete",
            "aggregate": agg,
            "by_category": by_cat,
            "cases": [
                {
                    "id": cid,
                    "category": cat,
                    "mode": turn.mode,
                    "hit_count": len(turn.hits or []),
                    "passed": sc.passed,
                    "notes": sc.notes,
                    "answer_preview": (turn.text or "")[:500],
                }
                for cid, cat, turn, sc in rows
            ],
        }
        args.json_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_report}")

    return 0 if agg.get("passed", 0) == agg.get("cases", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
