"""Regression: live-transcript style phrasing stays on the document path (intent + extraction)."""

from __future__ import annotations

from app.llm.conversation_context import build_conversation_retrieval_hints
from app.llm.deterministic_extraction import try_build_grounded_document_overview, try_extract_field_value_answer
from app.llm.query_intent import normalize_query_for_field_intent, user_expects_document_grounding
from app.retrieval.vector_store import RetrievedChunk
from app.services.chat_service import wants_no_retrieval_fastpath


def _hit(text: str, *, source_name: str = "spacex_profile.txt") -> RetrievedChunk:
    return RetrievedChunk(
        rank=0,
        page_content=text,
        metadata={"source_name": source_name, "chunk_id": "c1", "page_number": 1, "file_path": "data/raw/x.txt"},
        distance=0.9,
    )


def test_transcript_queries_expect_retrieval_not_fastpath() -> None:
    phrases = [
        "summarize this document",
        "summarize the document",
        "what company is discussed",
        "what is the full name",
        "what is the website",
        "what is the contact email",
        "what is the contact number",
        "what is phne number",
        "what technologies does spacex use",
        "what projects are mentioned",
        "does Elon Musk know python",
        "how many employees does SpaceX have",
    ]
    for q in phrases:
        assert user_expects_document_grounding(q), q
        assert not wants_no_retrieval_fastpath(q), q


def test_typo_normalize_for_field_intent() -> None:
    assert "phone" in normalize_query_for_field_intent("what is phne number").lower()


def test_contact_email_and_website_extract() -> None:
    hits = [
        _hit("Contact Email: hiring@spacex.com\nWebsite: https://www.spacex.com\nPhone: +1 (555) 123-4567"),
    ]
    em = try_extract_field_value_answer("what is the contact email?", hits)
    assert em is not None and "hiring@spacex.com" in em.answer
    web = try_extract_field_value_answer("what is the website?", hits)
    assert web is not None and "spacex.com" in web.answer.lower()


def test_summarize_document_matches_overview_heuristic() -> None:
    hits = [
        _hit("Full Name: Test User\nCompany: SpaceX\nProjects: Starship, Starlink"),
        _hit("Technologies: Raptor engines, lithium batteries\nEmployees: 13000"),
        _hit("Full Name: Test User\nRole: Engineer"),
    ]
    out = try_build_grounded_document_overview("summarize this document", hits)
    assert out is not None
    assert "[source" in out.answer.lower()


def test_starship_followup_gets_document_routing_hints() -> None:
    hist = [
        {"role": "user", "content": "What is this document about?"},
        {
            "role": "assistant",
            "content": "The document discusses SpaceX programs including Starship.",
            "grounded": True,
            "sources": [{"source_name": "overview.txt"}],
        },
    ]
    h = build_conversation_retrieval_hints("what is starship", hist)
    assert h.force_document_scoped_routing
    assert "overview.txt" in h.retrieval_query
