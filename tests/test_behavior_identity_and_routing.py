"""Behavior consistency: assistant identity, general concepts, follow-ups (no retrieval changes)."""

from __future__ import annotations

from app.llm.conversation_context import build_conversation_retrieval_hints, effective_user_expects_document_grounding
from app.llm.query_intent import (
    is_assistant_identity_question,
    is_general_short_concept_query,
    should_bypass_document_intent_for_query,
    user_expects_document_grounding,
)


def test_assistant_identity_questions_not_document_scoped() -> None:
    for q in (
        "who are you?",
        "What is your name",
        "who built you",
        "what is this app?",
    ):
        assert is_assistant_identity_question(q), q
        assert should_bypass_document_intent_for_query(q), q
        assert user_expects_document_grounding(q) is False, q


def test_general_role_concepts_not_document_scoped() -> None:
    for q in (
        "what is ML engineer",
        "what is an ML engineer?",
        "data engineer",
        "data scientist?",
        "what is a data scientist",
    ):
        assert is_general_short_concept_query(q), q
        assert user_expects_document_grounding(q) is False, q


def test_document_questions_still_expect_grounding() -> None:
    assert user_expects_document_grounding("what is this document about?") is True
    assert user_expects_document_grounding("who is named as CFO?") is True


def test_who_are_you_after_grounded_turn_not_forced_doc() -> None:
    hist = [
        {"role": "user", "content": "Summarize the policy for unit ZZ-1."},
        {
            "role": "assistant",
            "content": "It lists intake steps [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "policy_zz1.txt", "source_number": 1}],
        },
    ]
    h = build_conversation_retrieval_hints("who are you?", hist)
    assert h.force_document_scoped_routing is False
    assert effective_user_expects_document_grounding("who are you?", h) is False


def test_general_role_after_grounded_not_forced_doc() -> None:
    hist = [
        {"role": "user", "content": "What does the spec say about latency?"},
        {
            "role": "assistant",
            "content": "Target is four hundred ms [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "spec_latency.txt"}],
        },
    ]
    h = build_conversation_retrieval_hints("data scientist", hist)
    assert h.force_document_scoped_routing is False


def test_follow_up_skill_and_typos_still_merge() -> None:
    hist = [
        {"role": "user", "content": "Who is the applicant on the form?"},
        {
            "role": "assistant",
            "content": "The applicant is listed as Case A-1 [SOURCE 1].",
            "grounded": True,
            "sources": [{"source_name": "form_case.txt", "source_number": 1}],
        },
    ]
    for q in ("does he know python?", "doe he knows about python", "did they include any address?"):
        h = build_conversation_retrieval_hints(q, hist)
        assert h.force_document_scoped_routing is True, q
        assert "form_case" in h.retrieval_query.lower(), q


def test_pronoun_follow_up_still_merge() -> None:
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
