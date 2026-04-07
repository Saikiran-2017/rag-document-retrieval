"""Phase 30: edge-case hardening for intent, long-context slicing, and multi-file focus."""

from __future__ import annotations

from app.llm.query_intent import is_section_navigation_query, is_sparse_entity_lookup_query
from app.llm.generator import _slice_text_around_match
from app.retrieval.context_selection import select_generation_context
from app.retrieval.vector_store import RetrievedChunk


def _hit(src: str, idx: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=idx,
        page_content=text,
        metadata={"source_name": src, "chunk_id": f"{src}_{idx}", "rrf_score": 0.03},
        distance=1.2,
    )


def test_weird_phrasing_still_detects_section_intent() -> None:
    assert is_section_navigation_query("In appendix 2, what does it say about DR?")
    assert is_section_navigation_query("what does part 7 say about disaster recovery")


def test_lookup_catches_ids_and_dates() -> None:
    assert is_sparse_entity_lookup_query("What is the invoice ID?")
    assert is_sparse_entity_lookup_query("What is the effective date for the policy?")
    assert is_sparse_entity_lookup_query("Who is the owner?")


def test_long_context_slicing_keeps_tail_fact() -> None:
    body = "Intro " + ("filler " * 800) + "\nCFO name on record: Maria Chen.\n" + ("tail " * 400)
    out = _slice_text_around_match(body, "Who is the CFO?", max_chars=400)
    assert "Maria Chen" in out
    assert len(out) <= 450


def test_focus_source_name_prevents_cross_file_dilution() -> None:
    ranked = [
        _hit("a.txt", 0, "CFO name on record: Maria Chen."),
        _hit("b.txt", 1, "irrelevant finance filler"),
        _hit("a.txt", 2, "more a"),
    ]
    ctx = select_generation_context(
        ranked,
        mode="qa",
        top_k=3,
        nvec=10,
        broad_document_question=False,
        focus_source_name="a.txt",
    )
    assert all(h.metadata.get("source_name") == "a.txt" for h in ctx[:2])

