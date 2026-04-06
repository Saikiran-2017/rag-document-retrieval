#!/usr/bin/env python3
"""
Manual hybrid retrieval smoke test (mirrors chat_service retrieval + gate + context).

From repo root::

    set PYTHONPATH=.
    set KA_RETRIEVAL_DEBUG=1
    .venv\\Scripts\\python.exe scripts/debug_retrieval_smoke.py

Options::

    --skip-sync          Load existing FAISS under --faiss-dir without rebuild (no API key).
    --work-dir PATH      Shorthand: raw=PATH/raw, faiss=PATH/faiss (e.g. eval/_work).

When sync is required, a valid OPENAI_API_KEY is needed to embed. Placeholder keys fail fast.
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


def _placeholder_key_message() -> str | None:
    from app.env_loader import is_openai_key_placeholder

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return "OPENAI_API_KEY is not set."
    bad, why = is_openai_key_placeholder(key)
    if bad:
        return f"OPENAI_API_KEY is not usable ({why}). Use a real key in `.env.local`."
    return None


def _pipeline_report(
    *,
    store: object,
    faiss_dir: Path,
    query: str,
    nvec: int,
    top_k: int,
    k_pool_cap: int,
) -> dict:
    from app.llm.query_intent import (
        is_broad_document_overview_query,
        uses_relaxed_document_grounding_gate,
    )
    from app.llm.query_rewrite import rewrite_for_retrieval
    from app.retrieval.context_selection import (
        hybrid_pool_size,
        rerank_hybrid_hits,
        select_generation_context,
    )
    from app.retrieval.hybrid_retrieve import hybrid_retrieve, merge_hybrid_hit_pools
    from app.services import document_health

    broad = is_broad_document_overview_query(query)
    relaxed = uses_relaxed_document_grounding_gate(query)
    wide = broad or relaxed
    base_pool = hybrid_pool_size(nvec, int(top_k))
    if wide:
        k_pool = min(nvec, max(base_pool, 22))
        kv = min(nvec, max(32, min(48, nvec)))
    else:
        k_pool = min(k_pool_cap, nvec, base_pool)
        kv = min(28, nvec)

    rewritten = rewrite_for_retrieval(query)
    pool_hits = hybrid_retrieve(
        store,
        rewritten,
        k_final=k_pool,
        k_vector=kv,
        k_bm25=kv,
    )
    if wide:
        boost_q = (
            f"{rewritten} main sections themes introduction conclusion purpose overview "
            "key ideas summary"
        ).strip()
        if relaxed and not broad:
            boost_q = (
                f"{boost_q} performance latency p99 workload reliability metrics discussion"
            ).strip()
        pool_b = hybrid_retrieve(
            store,
            boost_q,
            k_final=k_pool,
            k_vector=kv,
            k_bm25=kv,
        )
        pool_hits = merge_hybrid_hit_pools(pool_hits, pool_b)

    ranked = rerank_hybrid_hits(pool_hits)
    n_after_rerank = len(ranked)
    trusted = document_health.filter_trusted_retrieval_hits(faiss_dir, ranked)
    doc_ok, gate_reason = document_health.explain_allow_document_grounding(
        faiss_dir, trusted, relaxed_doc_qa=relaxed
    )
    ctx = (
        select_generation_context(
            trusted,
            mode="qa",
            top_k=int(top_k),
            nvec=nvec,
            broad_document_question=broad or relaxed,
        )
        if doc_ok
        else []
    )

    def _rows(hits: list, lim: int = 8) -> list[dict]:
        out = []
        for i, h in enumerate(hits[:lim]):
            out.append(
                {
                    "i": i,
                    "source": str(h.metadata.get("source_name") or ""),
                    "l2": round(float(h.distance), 4),
                    "rrf": round(float(h.metadata.get("rrf_score", 0) or 0), 5),
                    "preview": ((h.page_content or "")[:120]).replace("\n", " "),
                }
            )
        return out

    return {
        "user_query": query,
        "rewritten_query": rewritten,
        "broad_overview": broad,
        "relaxed_doc_gate": relaxed,
        "wide_retrieval": wide,
        "pool_size": len(pool_hits),
        "after_rerank": n_after_rerank,
        "after_trust_filter": len(trusted),
        "context_chunks_selected": len(ctx),
        "doc_grounding_allowed": doc_ok,
        "grounding_gate_reason": gate_reason,
        "llm_would_get_context": bool(doc_ok and ctx),
        "top_pool_hits": _rows(pool_hits),
        "top_trusted_hits": _rows(trusted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid retrieval smoke (full pipeline)")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--faiss-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None, help="Use PATH/raw and PATH/faiss")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--k-pool", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Load existing index only; do not rebuild (works without API key if index exists).",
    )
    args = parser.parse_args()

    from app.env_loader import load_repo_dotenv
    from app.ingestion.loader import get_default_raw_dir
    from app.retrieval.vector_store import (
        DEFAULT_EMBEDDING_MODEL,
        faiss_index_files_exist,
        faiss_vector_count,
        get_default_faiss_folder,
    )
    from app.services import debug_service, index_service

    load_repo_dotenv(ROOT)

    if args.work_dir:
        wd = args.work_dir.resolve()
        raw_dir = wd / "raw"
        faiss_dir = wd / "faiss"
    else:
        raw_dir = (args.raw_dir or get_default_raw_dir()).resolve()
        faiss_dir = (args.faiss_dir or get_default_faiss_folder()).resolve()

    has_index = faiss_index_files_exist(faiss_dir)
    print("=== Retrieval smoke ===", file=sys.stderr)
    print(
        json.dumps(
            {
                "raw_dir": str(raw_dir),
                "faiss_dir": str(faiss_dir),
                "index_files_exist": has_index,
                "skip_sync": args.skip_sync,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            },
            indent=2,
        ),
        file=sys.stderr,
    )

    if args.skip_sync:
        if not has_index:
            print(
                "ERROR: --skip-sync but no FAISS files under faiss_dir. "
                "Build an index first (valid API key) or point --faiss-dir / --work-dir to one.",
                file=sys.stderr,
            )
            return 2
        store = index_service.load_faiss_store(faiss_dir)
    else:
        ph = _placeholder_key_message()
        if ph and not has_index:
            print(f"ERROR: {ph}", file=sys.stderr)
            print(
                "Hint: use a real key in .env.local, or build index elsewhere and re-run with "
                "--skip-sync --faiss-dir <path> (or --work-dir eval/_work after a successful eval).",
                file=sys.stderr,
            )
            return 3
        ok, msg = index_service.ensure_index_matches_library(
            raw_dir,
            faiss_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(
            json.dumps({"sync_ok": ok, "sync_message": (msg or "")[:500]}, indent=2),
            file=sys.stderr,
        )
        if not ok:
            print(
                "ERROR: index sync failed. If the key is invalid, fix OPENAI_API_KEY; "
                "or use --skip-sync with an existing index.",
                file=sys.stderr,
            )
            return 2
        store = index_service.load_faiss_store(faiss_dir)

    nvec = faiss_vector_count(store)
    print(json.dumps({"vector_count": nvec}, indent=2), file=sys.stderr)
    if nvec == 0:
        print("ERROR: empty vector index.", file=sys.stderr)
        return 2

    queries = (
        "CEO name Acme executive leadership",
        "Q3 revenue million finance",
        "ZEPHYR playbook retrieval internal",
        "p99 latency requirements milliseconds",
    )

    for q in queries:
        report = _pipeline_report(
            store=store,
            faiss_dir=faiss_dir,
            query=q,
            nvec=nvec,
            top_k=args.top_k,
            k_pool_cap=args.k_pool,
        )
        debug_service.log_retrieval_event("smoke_full_pipeline", **report)
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
