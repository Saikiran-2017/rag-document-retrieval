"""
Lightweight scoring for document QA evaluation (no extra ML frameworks).

Dimensions map to interview-friendly metrics: routing, refusal calibration, retrieval anchors,
answer keyword overlap, forbidden tokens (hallucination proxy).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.llm.generator import UNKNOWN_PHRASE
from app.retrieval.vector_store import RetrievedChunk
from app.services.chat_service import AssistantTurn


@dataclass
class CaseScores:
    """Per-dimension pass/fail for one gold case."""

    routing_ok: bool
    refusal_ok: bool
    false_refusal: bool
    answer_keywords_ok: bool
    retrieval_relevance_ok: bool
    forbidden_ok: bool
    citation_surface_ok: bool | None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        base = (
            self.routing_ok
            and self.refusal_ok
            and not self.false_refusal
            and self.answer_keywords_ok
            and self.retrieval_relevance_ok
            and self.forbidden_ok
        )
        if self.citation_surface_ok is False:
            return False
        return base


def _norm(s: str) -> str:
    return (s or "").lower()


def looks_like_refusal(text: str) -> bool:
    """Heuristic: model declined to answer from documents."""
    t = _norm(text)
    if _norm(UNKNOWN_PHRASE) in t:
        return True
    if "don't know" in t and "document" in t:
        return True
    if "cannot" in t and "document" in t and ("find" in t or "locate" in t or "provided" in t):
        return True
    if "not" in t and "provided" in t and "document" in t:
        return True
    if "do not contain" in t or "does not contain" in t or "don't contain" in t:
        return True
    if "no information" in t and ("document" in t or "file" in t or "upload" in t):
        return True
    return False


def _hits_blob(hits: list[RetrievedChunk] | None) -> str:
    if not hits:
        return ""
    parts = [(h.page_content or "") for h in hits]
    return _norm("\n".join(parts))


def _cited_source_numbers(answer: str) -> set[int]:
    return {int(m.group(1)) for m in re.finditer(r"\[SOURCE\s+(\d+)\]", answer, re.I)}


def score_case(
    turn: AssistantTurn,
    expected: dict[str, Any],
    *,
    require_citations_when_grounded: bool = True,
) -> CaseScores:
    """
    Compare ``turn`` to gold ``expected`` block from JSON.

    ``expected`` keys:
      mode_in, refusal, answer_substrings_any, retrieval_anchors_any,
      answer_substrings_must_not
    """
    exp = expected or {}
    mode_in = list(exp.get("mode_in") or ["grounded"])
    want_refusal = bool(exp.get("refusal"))
    any_kw = [_norm(x) for x in (exp.get("answer_substrings_any") or []) if str(x).strip()]
    anchors = [x for x in (exp.get("retrieval_anchors_any") or []) if str(x).strip()]
    forbidden = [_norm(x) for x in (exp.get("answer_substrings_must_not") or []) if str(x).strip()]

    text = turn.text or ""
    tnorm = _norm(text)
    hits = turn.hits or []

    routing_ok = turn.mode in mode_in

    got_refusal = looks_like_refusal(text)
    if want_refusal:
        # Abstain from fabricating; general/web without doc claims is acceptable if forbidden_ok passes.
        refusal_ok = got_refusal or turn.mode in ("general", "web")
    else:
        refusal_ok = not got_refusal

    false_refusal = (not want_refusal) and got_refusal

    if any_kw:
        answer_keywords_ok = any(k in tnorm for k in any_kw)
    else:
        answer_keywords_ok = True

    hblob = _hits_blob(hits)
    if anchors:
        retrieval_relevance_ok = any(_norm(a) in hblob for a in anchors)
    else:
        retrieval_relevance_ok = True

    forbidden_ok = all(f not in tnorm for f in forbidden)

    citation_surface_ok: bool | None = None
    if want_refusal:
        citation_surface_ok = None
    elif require_citations_when_grounded and turn.mode in ("grounded", "blended") and hits:
        cited = _cited_source_numbers(text)
        citation_surface_ok = bool(cited) and max(cited, default=0) <= len(hits)
    elif turn.mode in ("grounded", "blended") and not hits:
        citation_surface_ok = False

    notes: list[str] = []
    if not routing_ok:
        notes.append(f"routing: got {turn.mode!r}, expected one of {mode_in}")
    if not refusal_ok:
        notes.append(f"refusal: want_refusal={want_refusal}, got_refusal={got_refusal}, mode={turn.mode}")
    if false_refusal:
        notes.append("false_refusal: model abstained but gold expects a substantive grounded answer")
    if not answer_keywords_ok and any_kw:
        notes.append(f"answer_keywords: expected any of {any_kw!r} in answer")
    if not retrieval_relevance_ok and anchors:
        notes.append(f"retrieval_anchors: none of {anchors!r} found in retrieved hit bodies")
    if not forbidden_ok:
        notes.append(f"forbidden substring appeared from {forbidden!r}")
    if citation_surface_ok is False:
        notes.append("citations: grounded turn missing [SOURCE n] or cite out of range")

    return CaseScores(
        routing_ok=routing_ok,
        refusal_ok=refusal_ok,
        false_refusal=false_refusal,
        answer_keywords_ok=answer_keywords_ok,
        retrieval_relevance_ok=retrieval_relevance_ok,
        forbidden_ok=forbidden_ok,
        citation_surface_ok=citation_surface_ok,
        notes=notes,
    )


def aggregate_by_category(
    rows: list[tuple[str, str, AssistantTurn, CaseScores]],
) -> dict[str, dict[str, Any]]:
    """Group pass/fail and case detail by ``category`` (for eval reports)."""
    from collections import defaultdict

    out: dict[str, dict[str, Any]] = {}
    bucket: dict[str, list[tuple[str, AssistantTurn, CaseScores]]] = defaultdict(list)
    for cid, cat, turn, sc in rows:
        bucket[cat or "unknown"].append((cid, turn, sc))
    for cat, items in bucket.items():
        passed = sum(1 for _, _, s in items if s.passed)
        out[cat] = {
            "total": len(items),
            "passed": passed,
            "failed": len(items) - passed,
            "pass_rate": round(passed / len(items), 4) if items else 0.0,
            "cases": [
                {
                    "id": cid,
                    "passed": sc.passed,
                    "mode": getattr(turn, "mode", None),
                    "hit_count": len(getattr(turn, "hits", None) or []),
                    "notes": sc.notes,
                }
                for cid, turn, sc in items
            ],
        }
    return out


def aggregate_rates(results: list[tuple[str, CaseScores]]) -> dict[str, float | int]:
    """Roll up rates for the final report."""
    n = len(results)
    if not n:
        return {}
    false_ref = sum(1 for _, s in results if s.false_refusal)
    hall_proxy = sum(1 for _, s in results if not s.forbidden_ok)
    rel = sum(1 for _, s in results if s.retrieval_relevance_ok)
    rout = sum(1 for _, s in results if s.routing_ok)
    passed = sum(1 for _, s in results if s.passed)
    return {
        "cases": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4),
        "routing_ok_rate": round(rout / n, 4),
        "retrieval_anchor_rate": round(rel / n, 4),
        "false_refusal_count": false_ref,
        "forbidden_hit_count": hall_proxy,
    }
