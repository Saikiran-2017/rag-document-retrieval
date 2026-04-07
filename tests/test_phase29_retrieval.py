"""Phase 29: section prioritization, entity lookup intent, lookup hybrid gate."""

from __future__ import annotations

from app.llm.generator import hybrid_retrieval_is_useful
from app.services.chat_service import _wants_blended_web
from app.llm.query_intent import (
    is_section_navigation_query,
    is_sparse_entity_lookup_query,
)
from app.retrieval.context_selection import prioritize_section_navigation_hits
from app.retrieval.vector_store import RetrievedChunk


def _chunk(text: str, rank: int = 0, d: float = 1.0, rrf: float = 0.03) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        page_content=text,
        metadata={"rrf_score": rrf, "source_name": "t.txt"},
        distance=d,
    )


def test_prioritize_section_moves_disaster_anchor_up() -> None:
    filler = "### Appendix\nfiller " * 30
    anchor = "### Deep section\nPHASE28-ANCHOR: section seven discusses disaster recovery."
    hits = [_chunk(filler, rank=i) for i in range(4)] + [_chunk(anchor, rank=4)]
    q = "What does section seven or disaster recovery say?"
    out = prioritize_section_navigation_hits(hits, q)
    assert "PHASE28-ANCHOR" in out[0].page_content


def test_sparse_entity_lookup_matches_cfo_question() -> None:
    assert is_sparse_entity_lookup_query("Who is named as CFO or finance lead?")


def test_sparse_entity_lookup_rejects_eval_negative_queries() -> None:
    assert not is_sparse_entity_lookup_query(
        "According to my uploaded documents, what is the exact recipe for chocolate cake?"
    )
    assert not is_sparse_entity_lookup_query(
        "What year did humans land on Mars according to the internal playbook?"
    )


def test_section_nav_detects_disaster_recovery() -> None:
    assert is_section_navigation_query("What does the disaster recovery section cover?")


def test_entity_lookup_does_not_force_blended_web() -> None:
    assert not _wants_blended_web("Who is named as CFO or finance lead?")


def test_hybrid_lookup_allows_borderline_l2_with_strong_rrf() -> None:
    h = [_chunk("cfo maria", d=1.232, rrf=0.0325)]
    assert not hybrid_retrieval_is_useful(h)
    assert hybrid_retrieval_is_useful(h, for_lookup_qa=True)
