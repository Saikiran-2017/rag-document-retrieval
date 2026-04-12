#!/usr/bin/env python3
"""
Brutal high-level product check (no FastAPI required).

Upload (simulated by writing files) -> sync -> two grounded chat probes.

**Important:** uses a fresh raw + FAISS directory each run so a stale on-disk index
cannot make probes look like they pass while vectors are wrong.

If OPENAI_API_KEY is missing/placeholder, fails fast.

Exit codes:
  0 — success, both probes grounded with expected anchors
  2 — bad API key
  3 — sync failed
  4 — probe 1 not grounded or missing CFO anchor
  5 — probe 2 not grounded or missing section / DR anchor
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path


def main() -> int:
    # Match eval harness: non-streaming turns populate ``text`` immediately (anchors check ``t.text``).
    os.environ["KA_NO_STREAM"] = "1"
    os.environ.setdefault("WEB_SEARCH_ENABLED", "0")

    from app.env_loader import is_openai_key_placeholder
    from app.services import index_service
    from app.services.chat_service import answer_user_query

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    bad, why = is_openai_key_placeholder(key)
    if bad:
        print(
            "[brutal] FAIL: OPENAI_API_KEY missing/placeholder. "
            f"Set a real key in .env.local or your shell (detected: {why})."
        )
        return 2

    work = Path(tempfile.mkdtemp(prefix="ka_brutal_"))
    raw = work / "raw"
    faiss = work / "faiss"
    raw.mkdir(parents=True, exist_ok=True)
    faiss.mkdir(parents=True, exist_ok=True)

    (raw / "brief.txt").write_text(
        "# Reliability Brief\n\nCFO name on record: Maria Chen.\n\nSection 7 discusses disaster recovery.\n",
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    ok, msg, nvec, action = index_service.rebuild_knowledge_index(
        raw, faiss, chunk_size=900, chunk_overlap=120
    )
    print(
        f"[sync] ok={ok} action={action} vectors={nvec} took_s={time.perf_counter() - t0:.2f} msg={msg!r}"
    )
    if not ok:
        shutil.rmtree(work, ignore_errors=True)
        return 3
    if nvec <= 0:
        print("[brutal] FAIL: vector_count is zero after sync (library not indexed).")
        shutil.rmtree(work, ignore_errors=True)
        return 3
    if action not in ("rebuilt", "unchanged"):
        print(f"[brutal] FAIL: unexpected sync_action {action!r}")
        shutil.rmtree(work, ignore_errors=True)
        return 3

    q1 = "Who is named as CFO?"
    t1 = answer_user_query(q1, raw_dir=raw, faiss_folder=faiss, chunk_size=900, chunk_overlap=120, top_k=6)
    print(f"[chat] q={q1!r} mode={t1.mode} preview={t1.text[:160]!r}")
    if t1.mode != "grounded":
        print("[brutal] FAIL: expected grounded answer for CFO probe.")
        shutil.rmtree(work, ignore_errors=True)
        return 4
    low1 = t1.text.lower()
    if "maria" not in low1 and "chen" not in low1:
        print("[brutal] FAIL: CFO probe answer missing expected anchor tokens.")
        shutil.rmtree(work, ignore_errors=True)
        return 4

    q2 = "What does section 7 say?"
    t2 = answer_user_query(q2, raw_dir=raw, faiss_folder=faiss, chunk_size=900, chunk_overlap=120, top_k=6)
    print(f"[chat] q={q2!r} mode={t2.mode} preview={t2.text[:160]!r}")
    if t2.mode != "grounded":
        print("[brutal] FAIL: expected grounded answer for section 7 probe.")
        shutil.rmtree(work, ignore_errors=True)
        return 5
    low2 = t2.text.lower()
    if "disaster" not in low2 and "recovery" not in low2 and "section" not in low2:
        print("[brutal] FAIL: section 7 probe missing expected anchor tokens.")
        shutil.rmtree(work, ignore_errors=True)
        return 5

    shutil.rmtree(work, ignore_errors=True)
    print("[brutal] PASS: sync + grounded CFO + grounded section 7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
