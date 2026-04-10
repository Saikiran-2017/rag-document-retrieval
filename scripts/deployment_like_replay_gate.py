#!/usr/bin/env python3
"""
Deployment-like replay (non-Docker fallback):

- Starts the real FastAPI app via Uvicorn in a subprocess (HTTP server).
- Replays the same privacy-safe transcript pattern through the real HTTP routes:
    POST /api/v1/sync
    POST /api/v1/chat  (with accumulating `conversation`)

This is used when Docker Engine is unavailable on the machine. It validates the
real product HTTP path and produces a strict per-question report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

FIXTURE_NAME = "transcript_ref_alpha.txt"
FIXTURE_TEXT = """# Internal Unit Onboarding — Reference Sheet ALPHA

Purpose: summarize onboarding steps for internal unit ALPHA-9 before system access.

Subject Name: HOLDER BETA
Email: contact@example.invalid
Contact Number: +1 (555) 010-0199
Current Address: 100 Example Lane, Suite 5, Sample City, SC 12345

Section notes: facility badge pickup only. Passport identifiers are not collected in this form.
"""


def _die(msg: str, code: int = 2) -> None:
    print(f"[deployment_like_gate] FAIL: {msg}")
    raise SystemExit(code)


def _assistant_turn_from_response(resp: dict[str, Any]) -> dict[str, Any]:
    """Match Next.js / transcript gate: include sources + grounded flag for follow-up retrieval."""
    row: dict[str, Any] = {
        "role": "assistant",
        "content": str(resp.get("text") or ""),
        "mode": resp.get("mode"),
    }
    srcs = resp.get("sources")
    if isinstance(srcs, list) and srcs:
        row["sources"] = srcs
        row["grounded"] = True
    return row


def _wait_for_health(base_url: str, timeout_s: float = 60.0) -> None:
    t0 = time.perf_counter()
    last_err = ""
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return
            last_err = f"health HTTP {r.status_code}: {r.text[:200]!r}"
        except Exception as e:  # noqa: BLE001 - printed as diagnostic, then raised
            last_err = repr(e)
        time.sleep(0.5)
    _die(f"server health never became ready (last_err={last_err})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Deployment-like transcript replay via real HTTP server.")
    ap.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Existing API base URL (e.g. http://127.0.0.1:8000 for docker compose). "
        "If set, no local uvicorn subprocess is started; fixture is written to data/raw/.",
    )
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--json-out", type=str, default="eval/_deployment_like_gate_report.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    # Ensure we have the same env loading behavior as the app/scripts.
    load_dotenv(root / ".env.local", override=False)
    load_dotenv(root / ".env", override=False)

    use_external = bool((args.base_url or "").strip())
    proc: subprocess.Popen[str] | None = None
    work: Path | None = None

    if use_external:
        base_url = (args.base_url or "").strip().rstrip("/")
        raw_host = root / "data" / "raw"
        raw_host.mkdir(parents=True, exist_ok=True)
        (raw_host / FIXTURE_NAME).write_text(FIXTURE_TEXT, encoding="utf-8")
    else:
        # Fresh, private workdir so the sync is deterministic and avoids local residue.
        work = Path(tempfile.mkdtemp(prefix="ka_deploy_like_"))
        raw = work / "raw"
        faiss = work / "faiss"
        raw.mkdir(parents=True, exist_ok=True)
        faiss.mkdir(parents=True, exist_ok=True)
        (raw / FIXTURE_NAME).write_text(FIXTURE_TEXT, encoding="utf-8")

        base_url = f"http://127.0.0.1:{args.port}"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env.setdefault("KA_DEBUG", "1")
        env.setdefault("KA_NO_STREAM", "1")
        env["KA_ENV"] = env.get("KA_ENV") or "production"
        env["KA_RAW_DIR"] = str(raw)
        env["KA_FAISS_DIR"] = str(faiss)
        # Ensure API key is present in the server subprocess (common cause of hangs/timeouts).
        if not (env.get("OPENAI_API_KEY") or "").strip():
            _die("OPENAI_API_KEY not set in environment (set it in your shell or .env.local)")

        # Start server (real HTTP path).
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--log-level",
                "warning",
            ],
            cwd=str(root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        _wait_for_health(base_url, timeout_s=180.0 if use_external else 75.0)

        # 1) Sync
        sr = requests.post(f"{base_url}/api/v1/sync", json={}, timeout=120.0)
        if sr.status_code != 200:
            _die(f"sync HTTP {sr.status_code}: {sr.text[:500]}")
        sync_body = sr.json()
        if not sync_body.get("ok"):
            _die(f"sync ok=False: {sync_body}")
        if int(sync_body.get("vector_count") or 0) <= 0:
            _die("sync returned vector_count<=0")

        conversation: list[dict[str, Any]] = []
        report_rows: list[dict[str, Any]] = []
        # Shared host `data/raw` may contain multiple files; metadata answers follow the focused source.
        meta_name_tokens = (
            ["uploaded library file", ".txt"]
            if use_external
            else ["transcript_ref_alpha"]
        )
        address_tokens = (
            ["address", "[source"]
            if use_external
            else ["example lane", "address"]
        )

        def _infer_deterministic(routing: str, text: str) -> bool:
            r = (routing or "").lower()
            if r.startswith("grounded_deterministic"):
                return True
            tl = (text or "").lower()
            if "uploaded library file" in tl and "transcript_ref" in tl:
                return True
            if "name on record:" in tl:
                return True
            if "key items it contains" in tl or "structured administrative record" in tl:
                return True
            if "address on record:" in tl:
                return True
            return False

        def chat(message: str) -> dict[str, Any]:
            body: dict[str, Any] = {
                "message": message,
                "task_mode": "auto",
                "summarize_scope": "all",
            }
            if conversation:
                body["conversation"] = conversation
            # This route is non-streaming JSON, but model calls can still take time.
            cr = requests.post(f"{base_url}/api/v1/chat", json=body, timeout=600.0)
            if cr.status_code != 200:
                _die(f"chat HTTP {cr.status_code} for {message[:60]!r}: {cr.text[:500]}")
            return cr.json()

        def check_step(
            label: str,
            question: str,
            data: dict[str, Any],
            *,
            expect_mode: str,
            retrieval_min: int | None = None,
            must_contain: list[str] | None = None,
            must_not_contain: list[str] | None = None,
            citation: bool | None = None,
        ) -> None:
            mode = str(data.get("mode") or "")
            text = str(data.get("text") or "")
            diag = data.get("diagnostics") if isinstance(data.get("diagnostics"), dict) else {}
            routing = str(diag.get("routing") or diag.get("route_selected") or "")
            rcount = data.get("retrieval_chunk_count")
            r_ran = diag.get("retrieval_ran")
            cites = "[SOURCE" in text.upper()
            srcs = diag.get("selected_sources")
            dom = str(srcs[0]) if isinstance(srcs, list) and srcs else None

            report_rows.append(
                {
                    "scenario_label": label,
                    "user_question": question,
                    "route_selected": routing,
                    "retrieval_fired": bool(r_ran),
                    "retrieval_chunk_count": rcount,
                    "dominant_source_selected": dom,
                    "deterministic_extraction_fired": _infer_deterministic(routing, text),
                    "final_mode": mode,
                    "citations_present": cites,
                    "answer_correct": True,
                }
            )

            if mode != expect_mode:
                _die(f"{label}: expected mode={expect_mode!r}, got {mode!r}; preview={text[:220]!r}")
            if retrieval_min is not None:
                rc = int(rcount) if rcount is not None else 0
                if rc < retrieval_min:
                    _die(f"{label}: expected retrieval_chunk_count>={retrieval_min}, got {rc}")
            if citation is True and not cites and mode == "grounded":
                _die(f"{label}: expected [SOURCE n] in grounded answer; preview={text[:260]!r}")
            if citation is False and cites:
                _die(f"{label}: did not expect citations in answer")
            low = text.lower()
            for s in must_contain or []:
                if s.lower() not in low:
                    _die(f"{label}: expected substring {s!r}; preview={text[:320]!r}")
            for s in must_not_contain or []:
                if s.lower() in low:
                    _die(f"{label}: forbidden substring {s!r} appeared")

        def push_turn(user_msg: str, resp: dict[str, Any]) -> None:
            conversation.append({"role": "user", "content": user_msg})
            conversation.append(_assistant_turn_from_response(resp))

        # Scenarios (must match the required transcript pattern)
        m1 = "what is this document about?"
        d1 = chat(m1)
        check_step("broad_summary", m1, d1, expect_mode="grounded", retrieval_min=1, must_contain=["onboarding"], citation=True)
        push_turn(m1, d1)

        m2 = "what is his name?"
        d2 = chat(m2)
        check_step("follow_up_name", m2, d2, expect_mode="grounded", retrieval_min=1, must_contain=["holder", "beta"], citation=True)
        push_turn(m2, d2)

        m3 = "in document"
        d3 = chat(m3)
        check_step(
            "follow_up_in_document",
            m3,
            d3,
            expect_mode="grounded",
            retrieval_min=1,
            must_contain=["key items", "structured administrative"],
            citation=True,
        )
        push_turn(m3, d3)

        m4a = "what is the document name?"
        d4a = chat(m4a)
        check_step(
            "metadata_document_name",
            m4a,
            d4a,
            expect_mode="grounded",
            retrieval_min=1,
            must_contain=meta_name_tokens,
            citation=True,
        )
        push_turn(m4a, d4a)

        m4b = "what is the file name?"
        d4b = chat(m4b)
        check_step(
            "metadata_file_name",
            m4b,
            d4b,
            expect_mode="grounded",
            retrieval_min=1,
            must_contain=meta_name_tokens,
            citation=True,
        )
        push_turn(m4b, d4b)

        m5 = "did they include any address?"
        d5 = chat(m5)
        check_step(
            "field_address_inclusion",
            m5,
            d5,
            expect_mode="grounded",
            retrieval_min=1,
            must_contain=address_tokens,
            citation=True,
        )
        push_turn(m5, d5)

        m6 = "what is RAG"
        d6 = chat(m6)
        check_step("general_rag", m6, d6, expect_mode="general", must_contain=["retrieval", "augment"], citation=False)
        push_turn(m6, d6)

        m7 = "what is the passport number on file?"
        d7 = chat(m7)
        mode7 = str(d7.get("mode") or "")
        text7 = str(d7.get("text") or "").lower()
        diag7 = d7.get("diagnostics") if isinstance(d7.get("diagnostics"), dict) else {}
        routing7 = str(diag7.get("routing") or diag7.get("route_selected") or "")
        t7 = str(d7.get("text") or "")
        srcs7 = diag7.get("selected_sources")
        dom7 = str(srcs7[0]) if isinstance(srcs7, list) and srcs7 else None
        report_rows.append(
            {
                "scenario_label": "negative_missing_field",
                "user_question": m7,
                "route_selected": routing7,
                "retrieval_fired": bool(diag7.get("retrieval_ran")),
                "retrieval_chunk_count": d7.get("retrieval_chunk_count"),
                "dominant_source_selected": dom7,
                "deterministic_extraction_fired": routing7.lower().startswith("grounded_deterministic"),
                "final_mode": mode7,
                "citations_present": "[SOURCE" in t7.upper(),
                "answer_correct": True,
            }
        )
        if mode7 != "grounded":
            _die(f"negative_missing_field: expected mode=grounded, got {mode7!r}")
        if int(d7.get("retrieval_chunk_count") or 0) < 1:
            _die("negative_missing_field: expected retrieval_chunk_count>=1")
        if not any(
            p in text7
            for p in (
                "don't know",
                "do not know",
                "cannot find",
                "can't find",
                "not in",
                "does not contain",
                "no passport",
                "not collected",
                "provided documents",
            )
        ):
            _die(f"negative_missing_field: expected refusal phrasing; got {text7[:400]!r}")

        out_path = (root / args.json_out).resolve() if not Path(args.json_out).is_absolute() else Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")

        print("[deployment_like_gate] PASS — HTTP replay OK")
        print(json.dumps(report_rows, indent=2))
        return 0
    finally:
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

