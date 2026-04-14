#!/usr/bin/env python3
"""
Replay a privacy-safe multi-turn transcript through the real FastAPI stack.

Uses POST /api/v1/sync and POST /api/v1/chat with an accumulating ``conversation``
payload (same shape as the Next.js client).

Run from repository root::

    set PYTHONPATH=.
    set KA_DEBUG=1
    python scripts/transcript_product_gate.py

Requires a non-placeholder OPENAI_API_KEY.

**Why it can feel slow:** one sync (embeddings over your fixture) plus ~8 chat turns, each
calling the remote chat model serially. Typical wall time is roughly **2–8 minutes** depending
on API latency and model load—not a hang. ``pytest`` without this script stays fast (~seconds).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

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
    print(f"[transcript_gate] FAIL: {msg}")
    raise SystemExit(code)


def _assistant_turn_for_conversation(resp: dict[str, Any]) -> dict[str, Any]:
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


def main() -> int:
    import time

    t_all = time.perf_counter()
    ap = argparse.ArgumentParser(description="Transcript gate via FastAPI TestClient.")
    ap.add_argument(
        "--workdir",
        type=str,
        default="",
        help="Directory for raw/ and faiss/ subfolders (created fresh). Default: temp directory.",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write a JSON report of each step.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root)
    os.environ.setdefault("KA_DEBUG", "1")
    os.environ.setdefault("KA_NO_STREAM", "1")

    from app.env_loader import is_openai_key_placeholder
    from backend.app.core.config import Settings, get_settings
    from backend.app.main import app
    from starlette.testclient import TestClient

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    bad, why = is_openai_key_placeholder(key)
    if bad:
        _die(f"OPENAI_API_KEY missing or placeholder ({why})")

    if args.workdir:
        work = Path(args.workdir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        raw = work / "raw"
        fss = work / "faiss"
        if raw.exists():
            shutil.rmtree(raw)
        if fss.exists():
            shutil.rmtree(fss)
        raw.mkdir(parents=True)
        fss.mkdir(parents=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="ka_transcript_gate_"))
        raw = work / "raw"
        fss = work / "faiss"
        raw.mkdir(parents=True)
        fss.mkdir(parents=True)

    (raw / FIXTURE_NAME).write_text(FIXTURE_TEXT, encoding="utf-8")

    get_settings.cache_clear()
    base = get_settings()
    custom = Settings(
        raw_dir=raw,
        faiss_dir=fss,
        default_chunk_size=base.default_chunk_size,
        default_chunk_overlap=base.default_chunk_overlap,
        default_top_k=base.default_top_k,
    )
    app.dependency_overrides[get_settings] = lambda: custom

    client = TestClient(app)

    sr = client.post("/api/v1/sync", json={})
    if sr.status_code != 200:
        _die(f"sync HTTP {sr.status_code}: {sr.text}")
    sync_body = sr.json()
    if not sync_body.get("ok"):
        _die(f"sync ok=False: {sync_body}")
    if int(sync_body.get("vector_count") or 0) <= 0:
        _die("sync returned vector_count<=0")

    report_rows: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []

    def chat(message: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "message": message,
            "task_mode": "auto",
            "summarize_scope": "all",
        }
        if conversation:
            body["conversation"] = conversation
        cr = client.post("/api/v1/chat", json=body)
        if cr.status_code != 200:
            _die(f"chat HTTP {cr.status_code} for {message[:60]!r}: {cr.text}")
        return cr.json()

    def _infer_deterministic(routing: str, text: str) -> bool:
        r = routing.lower()
        if r.startswith("grounded_deterministic"):
            return True
        tl = text.lower()
        if "uploaded library file" in tl and "transcript_ref" in tl:
            return True
        if "name on record:" in tl:
            return True
        if "key items it contains" in tl or "structured administrative record" in tl:
            return True
        if "address on record:" in tl:
            return True
        return False

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
        dom = None
        if isinstance(srcs, list) and srcs:
            dom = str(srcs[0])

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
            _die(f"{label}: expected mode={expect_mode!r}, got {mode!r}; text preview={text[:220]!r}")
        if retrieval_min is not None:
            rc = int(rcount) if rcount is not None else 0
            if rc < retrieval_min:
                _die(f"{label}: expected retrieval_chunk_count>={retrieval_min}, got {rc}")
        if citation is True and not cites and mode == "grounded":
            _die(f"{label}: expected [SOURCE n] in grounded answer; text preview={text[:260]!r}")
        if citation is False and cites:
            _die(f"{label}: did not expect citations in answer")
        low = text.lower()
        for s in must_contain or []:
            if s.lower() not in low:
                _die(f"{label}: expected substring {s!r} in answer; got {text[:320]!r}")
        for s in must_not_contain or []:
            if s.lower() in low:
                _die(f"{label}: forbidden substring {s!r} appeared in answer")

    # 1) Broad summary
    m1 = "what is this document about?"
    d1 = chat(m1)
    check_step(
        "broad_summary",
        m1,
        d1,
        expect_mode="grounded",
        retrieval_min=1,
        must_contain=["onboarding"],
        citation=True,
    )
    conversation.append({"role": "user", "content": m1})
    conversation.append(_assistant_turn_for_conversation(d1))

    # 2) Follow-up pronoun
    m2 = "what is his name?"
    d2 = chat(m2)
    check_step(
        "follow_up_name",
        m2,
        d2,
        expect_mode="grounded",
        retrieval_min=1,
        must_contain=["holder", "beta"],
        citation=True,
    )
    conversation.append({"role": "user", "content": m2})
    conversation.append(_assistant_turn_for_conversation(d2))

    # 3) Short document-scoped follow-up
    m3 = "in document"
    d3 = chat(m3)
    check_step(
        "follow_up_in_document",
        m3,
        d3,
        expect_mode="grounded",
        retrieval_min=1,
        # Deictic follow-up uses deterministic overview bullets; corpus mix may omit specific field labels.
        must_contain=["key items", "structured administrative"],
        citation=True,
    )
    conversation.append({"role": "user", "content": m3})
    conversation.append(_assistant_turn_for_conversation(d3))

    # 4) Metadata
    m4a = "what is the document name?"
    d4a = chat(m4a)
    check_step(
        "metadata_document_name",
        m4a,
        d4a,
        expect_mode="grounded",
        retrieval_min=1,
        must_contain=["transcript_ref_alpha"],
        citation=True,
    )
    conversation.append({"role": "user", "content": m4a})
    conversation.append(_assistant_turn_for_conversation(d4a))

    m4b = "what is the file name?"
    d4b = chat(m4b)
    check_step(
        "metadata_file_name",
        m4b,
        d4b,
        expect_mode="grounded",
        retrieval_min=1,
        must_contain=["transcript_ref_alpha"],
        citation=True,
    )
    conversation.append({"role": "user", "content": m4b})
    conversation.append(_assistant_turn_for_conversation(d4b))

    # 5) Address inclusion
    m5 = "did they include any address?"
    d5 = chat(m5)
    check_step(
        "field_address_inclusion",
        m5,
        d5,
        expect_mode="grounded",
        retrieval_min=1,
        must_contain=["example lane", "address"],
        citation=True,
    )
    conversation.append({"role": "user", "content": m5})
    conversation.append(_assistant_turn_for_conversation(d5))

    # 6) General knowledge (no citations)
    m6 = "what is RAG"
    d6 = chat(m6)
    check_step(
        "general_rag",
        m6,
        d6,
        expect_mode="general",
        must_contain=["retrieval", "augment"],
        citation=False,
    )
    conversation.append({"role": "user", "content": m6})
    conversation.append(_assistant_turn_for_conversation(d6))

    # 7) Negative — passport not in doc
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
        _die(f"negative_passport: expected mode=grounded, got {mode7!r}")
    if int(d7.get("retrieval_chunk_count") or 0) < 1:
        _die("negative_passport: expected retrieval_chunk_count>=1")
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
            "provided documents",
        )
    ):
        _die(f"negative_passport: expected refusal phrasing; got {text7[:400]!r}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report_rows, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t_all
    print(f"[transcript_gate] PASS — all transcript steps OK (wall time {elapsed:.1f}s)")
    print(json.dumps(report_rows, indent=2))
    app.dependency_overrides.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
