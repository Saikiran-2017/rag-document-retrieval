#!/usr/bin/env python3
"""
Phase 12 evaluation harness: hybrid retrieval + rerank + confidence gate (+ optional LLM).

Usage (from repo root, with OPENAI_API_KEY set and index built):

  python scripts/run_phase12_eval.py
  python scripts/run_phase12_eval.py --questions eval/phase12_questions.json
  python scripts/run_phase12_eval.py --with-llm   # slower; calls chat for each question

Exit code 0 always unless --strict is set and checks fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 12 retrieval eval harness")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "eval" / "phase12_questions.json",
    )
    parser.add_argument("--with-llm", action="store_true", help="Run full grounded generation (API calls)")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if expectation mismatches occur")
    args = parser.parse_args()

    from app.ingestion.loader import get_default_raw_dir
    from app.llm.generator import (
        UNKNOWN_PHRASE,
        generate_grounded_answer,
        hybrid_retrieval_is_useful,
    )
    from app.retrieval.context_selection import (
        hybrid_pool_size,
        rerank_hybrid_hits,
        select_generation_context,
    )
    from app.retrieval.hybrid_retrieve import hybrid_retrieve
    from app.retrieval.vector_store import (
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_INDEX_NAME,
        faiss_vector_count,
        get_default_faiss_folder,
        load_faiss_index,
    )
    from app.services.index_service import cached_openai_embeddings

    data = json.loads(args.questions.read_text(encoding="utf-8"))
    items = data.get("questions") or []

    raw_dir = get_default_raw_dir()
    faiss_folder = get_default_faiss_folder()
    embed = cached_openai_embeddings(DEFAULT_EMBEDDING_MODEL)
    store = load_faiss_index(
        folder_path=faiss_folder,
        index_name=DEFAULT_INDEX_NAME,
        embeddings=embed,
    )
    nvec = faiss_vector_count(store)
    print(f"Index vectors: {nvec}  raw_dir={raw_dir}  faiss={faiss_folder}\n")

    mismatches = 0
    top_k = 4
    for row in items:
        qid = row.get("id", "?")
        q = str(row.get("query", "")).strip()
        expect_g = bool(row.get("expect_doc_grounded", True))
        kws = [str(k).lower() for k in (row.get("expected_keywords") or []) if k]

        kp = hybrid_pool_size(nvec, top_k)
        pool = hybrid_retrieve(store, q, k_final=min(nvec, kp), k_vector=min(28, nvec), k_bm25=min(28, nvec))
        ranked = rerank_hybrid_hits(pool)
        useful = hybrid_retrieval_is_useful(ranked)
        ctx = select_generation_context(ranked, mode="qa", top_k=top_k, nvec=nvec) if useful else []

        line = (
            f"[{qid}] useful={useful} pool={len(pool)} ctx={len(ctx)} "
            f"best_d={float(ranked[0].distance) if ranked else None} "
            f"rrf={float(ranked[0].metadata.get('rrf_score', 0)) if ranked else None}"
        )
        print(line)
        print(f"      Q: {q[:100]}{'…' if len(q) > 100 else ''}")

        if expect_g != useful:
            print(f"      !! EXPECTATION_MISMATCH: expect_doc_grounded={expect_g} vs useful={useful}")
            mismatches += 1

        if args.with_llm and useful and ctx:
            ga = generate_grounded_answer(q, ctx)
            ok_unknown = UNKNOWN_PHRASE.lower() in ga.answer.lower()
            print(f"      A_preview: {ga.answer[:180]}{'…' if len(ga.answer) > 180 else ''}")
            if ga.validation_warning:
                print(f"      validation: {ga.validation_warning}")
            if kws and not ok_unknown:
                low = ga.answer.lower()
                if not any(k in low for k in kws):
                    print(f"      !! KEYWORD_MISS: expected one of {kws}")
                    mismatches += 1
        print()

    print(f"Done. mismatches={mismatches}")
    return 1 if args.strict and mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
