"""
Build a temp corpus from eval fixtures, index once, run gold cases through chat_service.

Run via ``python scripts/run_document_qa_eval.py`` from the repository root with
``OPENAI_API_KEY`` set and a real venv (FAISS + embeddings).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

from app.services import chat_service
from app.services.chat_service import AssistantTurn
from app.services.index_service import rebuild_knowledge_index
from app.services.message_service import DEFAULT_TOP_K

from eval.scoring import CaseScores, aggregate_rates, score_case


EVAL_ROOT = Path(__file__).resolve().parent


@dataclass
class EvalConfig:
    version: int
    description: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    corpus_files: list[str]
    cases: list[dict[str, Any]]


def load_config(path: Path | None = None) -> EvalConfig:
    p = path or (EVAL_ROOT / "gold_cases.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    return EvalConfig(
        version=int(data.get("version", 1)),
        description=str(data.get("description", "")),
        chunk_size=int(data["chunk_size"]),
        chunk_overlap=int(data["chunk_overlap"]),
        top_k=int(data.get("top_k", DEFAULT_TOP_K)),
        corpus_files=list(data["corpus_files"]),
        cases=list(data["cases"]),
    )


def prepare_corpus(raw_dir: Path, cfg: EvalConfig) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for rel in cfg.corpus_files:
        src = EVAL_ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"Fixture missing: {src}")
        dst = raw_dir / Path(rel).name
        shutil.copy2(src, dst)


def run_eval(
    *,
    cfg_path: Path | None = None,
    work_dir: Path | None = None,
) -> Iterator[tuple[str, str, AssistantTurn, CaseScores]]:
    """
    Yields ``(case_id, category, turn, scores)`` per gold case.
    Sets process env for deterministic routing (no streaming, no web).
    """
    cfg = load_config(cfg_path)
    base = work_dir or (EVAL_ROOT / "_work")
    raw_dir = base / "raw"
    faiss_dir = base / "faiss"
    if base.exists():
        shutil.rmtree(base)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    prepare_corpus(raw_dir, cfg)

    os.environ["KA_NO_STREAM"] = "1"
    os.environ["WEB_SEARCH_ENABLED"] = "0"

    ok, msg, nvec, action = rebuild_knowledge_index(
        raw_dir,
        faiss_dir,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    if not ok:
        raise RuntimeError(f"Index build failed: {msg} (action={action})")
    if nvec < 1:
        raise RuntimeError("Index build produced zero vectors")

    for case in cfg.cases:
        cid = str(case["id"])
        category = str(case.get("category", ""))
        query = str(case["query"])
        task_mode = str(case.get("task_mode", "auto"))
        expected = dict(case.get("expected") or {})

        turn = chat_service.answer_user_query(
            query,
            raw_dir=raw_dir,
            faiss_folder=faiss_dir,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            top_k=cfg.top_k,
            task_mode=cast(Any, task_mode),
        )
        if turn.stream_tokens is not None:
            raise RuntimeError("Streaming should be disabled (KA_NO_STREAM) for eval")

        sc = score_case(turn, expected)
        yield cid, category, turn, sc


def format_report(rows: list[tuple[str, str, AssistantTurn, CaseScores]]) -> str:
    lines: list[str] = []
    lines.append("=== Document QA eval report ===\n")
    agg = aggregate_rates([(a, d) for a, _, _, d in rows])
    lines.append(json.dumps(agg, indent=2))
    lines.append("")
    for cid, cat, turn, sc in rows:
        status = "PASS" if sc.passed else "FAIL"
        lines.append(f"[{status}] {cid} ({cat})  mode={turn.mode}  hits={len(turn.hits or [])}")
        preview = (turn.text or "").replace("\n", " ")[:160]
        if preview:
            lines.append(f"  answer: {preview}…")
        for n in sc.notes:
            lines.append(f"  ! {n}")
        lines.append("")
    return "\n".join(lines)
