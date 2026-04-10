from __future__ import annotations

from app.llm.deterministic_extraction import try_answer_section_navigation_fallback
from app.retrieval.vector_store import RetrievedChunk


def _hit(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=0,
        page_content=text,
        metadata={"source_name": "brief.txt", "chunk_id": "c1"},
        distance=0.95,
    )


def test_section_number_sentence_extracted() -> None:
    hits = [_hit("CFO on record: Alpha. Section 7 discusses disaster recovery. Footer text.")]
    out = try_answer_section_navigation_fallback("What does section 7 say?", hits)
    assert out is not None
    assert "disaster" in out.answer.lower()
    assert "[SOURCE 1]" in out.answer


def test_non_section_query_returns_none() -> None:
    hits = [_hit("Section 7 discusses disaster recovery.")]
    assert try_answer_section_navigation_fallback("who is the CFO?", hits) is None
