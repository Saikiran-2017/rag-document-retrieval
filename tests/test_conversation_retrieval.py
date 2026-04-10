"""Conversation-aware retrieval hints (follow-ups, prior grounded sources)."""

from __future__ import annotations

from app.llm.conversation_context import build_conversation_retrieval_hints, effective_user_expects_document_grounding


def test_follow_up_merges_prior_user_query_and_dominant_source() -> None:
    hist = [
        {"role": "user", "content": "What is in the uploaded policy packet for unit ALPHA-9?"},
        {
            "role": "assistant",
            "content": "It covers intake steps and contact fields [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "fixture_policy_alpha.txt", "source_number": 1}],
        },
    ]
    h = build_conversation_retrieval_hints("what is his name?", hist)
    assert h.force_document_scoped_routing is True
    assert h.focus_source_name == "fixture_policy_alpha.txt"
    assert "fixture_policy_alpha" in h.retrieval_query.lower()
    assert "ALPHA-9" in h.retrieval_query or "alpha-9" in h.retrieval_query.lower()


def test_general_tech_question_not_treated_as_doc_follow_up() -> None:
    hist = [
        {"role": "user", "content": "What does the spec say about latency?"},
        {
            "role": "assistant",
            "content": "Target is four hundred ms [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "spec_latency.txt"}],
        },
    ]
    h = build_conversation_retrieval_hints("what is RAG?", hist)
    assert h.force_document_scoped_routing is False
    assert h.retrieval_query.strip() == "what is RAG?"


def test_effective_document_intent_includes_conversation_override() -> None:
    hist = [
        {
            "role": "assistant",
            "content": "Field X is twelve [SOURCE 1].",
            "sources": [{"source_name": "ledger_a.txt"}],
        },
    ]
    h = build_conversation_retrieval_hints("did they include any address?", hist)
    assert h.force_document_scoped_routing is True
    assert effective_user_expects_document_grounding("did they include any address?", h) is True


def test_in_document_without_article_triggers_follow_up_routing() -> None:
    hist = [
        {"role": "user", "content": "Summarize the intake form for unit ZZ-1."},
        {
            "role": "assistant",
            "content": "It lists subject line items [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "intake_zz1.txt"}],
        },
    ]
    h = build_conversation_retrieval_hints("in document", hist)
    assert h.force_document_scoped_routing is True
    assert "intake_zz1" in h.retrieval_query


def test_metadata_question_triggers_follow_up_routing() -> None:
    hist = [
        {"role": "user", "content": "Open the intake form excerpt."},
        {
            "role": "assistant",
            "content": "It lists several labeled rows [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "intake_form_mock.txt"}],
        },
    ]
    h = build_conversation_retrieval_hints("what is the file name?", hist)
    assert h.force_document_scoped_routing is True
    assert h.focus_source_name == "intake_form_mock.txt"
