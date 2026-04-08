#!/usr/bin/env python3
"""
Brutal high-level product check (no FastAPI required).

Runs the same domain services as production: upload (simulated by writing files) -> sync -> chat.
If OPENAI_API_KEY is missing/placeholder, it will FAIL FAST with a clear message.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def main() -> int:
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

    base = Path("eval/_phase32_brutal_work").resolve()
    raw = base / "raw"
    faiss = base / "faiss"
    raw.mkdir(parents=True, exist_ok=True)
    faiss.mkdir(parents=True, exist_ok=True)

    # Minimal “real-ish” doc set.
    (raw / "brief.txt").write_text(
        "# Reliability Brief\n\nCFO name on record: Maria Chen.\n\nSection 7 discusses disaster recovery.\n",
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    ok, msg, nvec, action = index_service.rebuild_knowledge_index(raw, faiss, chunk_size=900, chunk_overlap=120)
    print(f"[sync] ok={ok} action={action} vectors={nvec} took_s={time.perf_counter()-t0:.2f} msg={msg!r}")
    if not ok:
        return 3

    q1 = "Who is named as CFO?"
    t1 = answer_user_query(q1, raw_dir=raw, faiss_folder=faiss, chunk_size=900, chunk_overlap=120, top_k=6)
    print(f"[chat] q={q1!r} mode={t1.mode} preview={t1.text[:120]!r}")

    q2 = "What does section 7 say?"
    t2 = answer_user_query(q2, raw_dir=raw, faiss_folder=faiss, chunk_size=900, chunk_overlap=120, top_k=6)
    print(f"[chat] q={q2!r} mode={t2.mode} preview={t2.text[:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

