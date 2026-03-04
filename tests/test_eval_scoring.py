"""Unit tests for eval scoring (no OpenAI / FAISS)."""

from __future__ import annotations

from app.llm.generator import GroundedAnswer, UNKNOWN_PHRASE
from app.retrieval.vector_store import RetrievedChunk
from app.services.chat_service import AssistantTurn

from eval.scoring import (
    CaseScores,
    aggregate_by_category,
    aggregate_rates,
    looks_like_refusal,
    score_case,
)


def test_looks_like_refusal_unknown_phrase():
    assert looks_like_refusal(UNKNOWN_PHRASE)


def test_looks_like_refusal_do_not_contain():
    assert looks_like_refusal("The documents do not contain a cake recipe.")


def test_score_case_routing_and_keywords():
    hits = [
        RetrievedChunk(
            rank=0,
            page_content="ZEPHYR-7 and IngestPlane are described here.",
            metadata={"source_name": "playbook_long.txt", "chunk_id": "x_0"},
            distance=0.5,
        )
    ]
    turn = AssistantTurn(
        mode="grounded",
        text="The playbook describes ZEPHYR-7 and IngestPlane [SOURCE 1].",
        grounded=GroundedAnswer(answer="x", sources=tuple()),
        hits=hits,
    )
    exp = {
        "mode_in": ["grounded"],
        "refusal": False,
        "answer_substrings_any": ["zephyr"],
        "retrieval_anchors_any": ["IngestPlane"],
        "answer_substrings_must_not": [],
    }
    sc = score_case(turn, exp)
    assert sc.routing_ok and sc.answer_keywords_ok and sc.retrieval_relevance_ok
    assert not sc.false_refusal
    assert sc.passed


def test_score_case_false_refusal():
    turn = AssistantTurn(mode="grounded", text=UNKNOWN_PHRASE, hits=[])
    exp = {"mode_in": ["grounded"], "refusal": False, "answer_substrings_any": ["zephyr"]}
    sc = score_case(turn, exp)
    assert sc.false_refusal
    assert not sc.passed


def test_aggregate_rates():
    rows = [
        ("a", CaseScores(True, True, False, True, True, True, None, [])),
        ("b", CaseScores(True, True, True, True, True, True, None, [])),
    ]
    agg = aggregate_rates(rows)
    assert agg["cases"] == 2
    assert agg["false_refusal_count"] == 1


def test_aggregate_by_category():
    turn = AssistantTurn(mode="grounded", text="x", hits=[])
    sc = CaseScores(True, True, False, True, True, True, None, [])
    rows = [("c1", "narrow_factual", turn, sc)]
    bc = aggregate_by_category(rows)
    assert bc["narrow_factual"]["passed"] == 1
    assert bc["narrow_factual"]["total"] == 1
