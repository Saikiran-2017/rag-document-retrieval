"""Intent continuity for short multi-turn follow-ups."""

from __future__ import annotations

from app.llm.query_normalize import normalize_query_for_pipeline
from app.llm.conversation_context import build_conversation_retrieval_hints


def test_and_phone_and_website_normalize_to_field_queries() -> None:
    assert normalize_query_for_pipeline("and phone number") == "what is the phone number"
    assert normalize_query_for_pipeline("and phone") == "what is the phone number"
    assert normalize_query_for_pipeline("and website") == "what is the website"
    assert normalize_query_for_pipeline("what about website") == "what is the website"


def test_confirmation_followup_forces_doc_routing_when_context_active() -> None:
    hist = [
        {"role": "user", "content": "what is the email"},
        {
            "role": "assistant",
            "content": "The document states that the email is a@b.co [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "one.txt", "source_number": 1}],
        },
    ]
    h = build_conversation_retrieval_hints("yes", hist)
    assert h.force_document_scoped_routing is True


def test_tell_me_more_after_summary_forces_doc_routing() -> None:
    """Continuation expansion query after grounded summary should force doc-scoped routing."""
    hist = [
        {"role": "user", "content": "summarize this document"},
        {
            "role": "assistant",
            "content": "This document contains information about company structure and projects [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "company.txt", "source_number": 1}],
        },
    ]
    h = build_conversation_retrieval_hints("tell me more", hist)
    assert h.force_document_scoped_routing is True
    assert h.focus_source_name == "company.txt"


def test_details_continuation_after_field_answer() -> None:
    """'details' after field lookup should force doc scoping with focus source."""
    hist = [
        {"role": "user", "content": "what is the email"},
        {
            "role": "assistant",
            "content": "The document states that the email is contact@company.com [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "profile.txt", "source_number": 1}],
        },
    ]
    h = build_conversation_retrieval_hints("details", hist)
    assert h.force_document_scoped_routing is True
    assert h.focus_source_name == "profile.txt"


def test_continue_expansion_preserves_focus_source() -> None:
    """'continue' after grounded answer should preserve document focus source."""
    hist = [
        {"role": "user", "content": "what projects has this company done"},
        {
            "role": "assistant",
            "content": "The company has worked on projects X and Y [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "experience.txt", "source_number": 1}],
        },
    ]
    h = build_conversation_retrieval_hints("continue", hist)
    assert h.force_document_scoped_routing is True
    assert h.focus_source_name == "experience.txt"


def test_elaborate_continuation_forces_scoped_retrieval() -> None:
    """'elaborate' expansion query should force document scoping."""
    hist = [
        {"role": "user", "content": "tell me about the team"},
        {
            "role": "assistant",
            "content": "The team includes senior engineers and product managers [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "team.txt", "source_number": 1}],
        },
    ]
    h = build_conversation_retrieval_hints("elaborate", hist)
    assert h.force_document_scoped_routing is True

