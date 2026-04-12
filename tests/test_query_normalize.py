"""Centralized pipeline query normalization (typos, light paraphrase, conservative fuzzy)."""

from __future__ import annotations

from app.llm.query_intent import (
    is_assistant_identity_question,
    is_broad_document_overview_query,
    is_general_short_concept_query,
    user_expects_document_grounding,
    uses_relaxed_document_grounding_gate,
)
from app.llm.query_normalize import normalize_query_for_pipeline
from app.services.chat_service import wants_no_retrieval_fastpath


def _n(q: str) -> str:
    return normalize_query_for_pipeline(q)


def test_summary_and_explain_typos_normalize() -> None:
    assert "summarize" in _n("sumarize this document")
    assert "summarize" in _n("summarise this document")
    assert "explain this document" in _n("explain this doc")
    assert is_broad_document_overview_query("sumarize this document")
    assert is_broad_document_overview_query("summarise this document")


def test_field_typos_and_contact_phrases() -> None:
    assert "email" in _n("what is the emial")
    assert "email" in _n("what is the contact email")
    assert "phone number" in _n("what is the phne number")
    assert "address" in _n("what is the adress")
    assert "website" in _n("what is the webiste")
    for q in (
        "what is the email",
        "what is the emial",
        "what is the contact email",
        "what is the phone number",
        "what is the phne number",
        "what is the address",
        "what is the adress",
        "what is the website",
        "what is the webiste",
    ):
        assert user_expects_document_grounding(q), q
        assert not wants_no_retrieval_fastpath(q), q


def test_routing_followup_phrases() -> None:
    assert "does he know python" in _n("doe he know python")
    assert "what company is discussed" in _n("what company is this about")
    assert "projects" in _n("what projects are mentioned").lower()
    assert "technologies" in _n("what technologies are used").lower()
    assert uses_relaxed_document_grounding_gate("what company is this about")
    assert user_expects_document_grounding("does Elon Musk know python")
    assert user_expects_document_grounding("what projects are mentioned")


def test_no_regression_identity_refusals_and_general_ml() -> None:
    for q in (
        "Who are you?",
        "What is your name?",
        "What model are you?",
    ):
        assert is_assistant_identity_question(q), q
        assert not user_expects_document_grounding(q), q
    for q in ("what is machine learning?", "what is ml?"):
        assert is_general_short_concept_query(q), q
        assert not user_expects_document_grounding(q), q
    assert wants_no_retrieval_fastpath("Hello there")
