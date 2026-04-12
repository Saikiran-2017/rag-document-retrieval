"""Document-scope heuristics: avoid false general fast paths on doc-shaped questions."""

from __future__ import annotations

from app.llm.query_intent import (
    is_broad_document_overview_query,
    is_general_short_concept_query,
    is_section_navigation_query,
    is_sparse_entity_lookup_query,
    user_expects_document_grounding,
    uses_relaxed_document_grounding_gate,
)
from app.services.chat_service import wants_no_retrieval_fastpath


def test_main_points_triggers_retrieval_not_fastpath():
    assert user_expects_document_grounding("What are the main points?")
    assert not wants_no_retrieval_fastpath("What are the main points?")


def test_main_topics_triggers_retrieval():
    assert user_expects_document_grounding("What are the main topics in this file?")
    assert not wants_no_retrieval_fastpath("What are the main topics in this file?")


def test_casual_greeting_can_fastpath():
    assert wants_no_retrieval_fastpath("Hello there")


def test_structured_field_questions_expect_document_grounding():
    for q in ("What is the email?", "what is his phone number?", "What is the full name?"):
        assert user_expects_document_grounding(q), q
        assert not wants_no_retrieval_fastpath(q), q


def test_machine_learning_short_query_stays_general_concept():
    for q in ("what is machine learning?", "machine learning", "what is ml?"):
        assert is_general_short_concept_query(q), q
        assert not user_expects_document_grounding(q), q


def test_broad_overview_flag():
    assert is_broad_document_overview_query("What is this document about?")
    assert is_broad_document_overview_query("Summarize this file")
    assert is_broad_document_overview_query("summarize this document")
    assert is_broad_document_overview_query("give me a summary of this document")
    assert not is_broad_document_overview_query("What is the revenue in Q3 per the table?")


def test_relaxed_gate_gold_broad_and_ambiguous():
    assert uses_relaxed_document_grounding_gate(
        "What is this playbook mainly about, in plain language?"
    )
    assert uses_relaxed_document_grounding_gate(
        "Give a concise summary of the key ideas in the long internal playbook."
    )
    assert uses_relaxed_document_grounding_gate("How is performance discussed?")
    assert uses_relaxed_document_grounding_gate("what company is discussed")
    assert uses_relaxed_document_grounding_gate("what projects are mentioned")
    assert uses_relaxed_document_grounding_gate("what technologies does spacex use")


def test_relaxed_gate_false_for_negative_eval_queries():
    assert not uses_relaxed_document_grounding_gate(
        "According to my uploaded documents, what is the exact recipe for chocolate cake?"
    )
    assert not uses_relaxed_document_grounding_gate(
        "What year did humans land on Mars according to the internal playbook?"
    )


def test_phase29_lookup_not_relaxed_for_negative_queries():
    assert not is_sparse_entity_lookup_query(
        "According to my uploaded documents, what is the exact recipe for chocolate cake?"
    )
    assert not is_section_navigation_query(
        "What year did humans land on Mars according to the internal playbook?"
    )


def test_eval_gold_queries_expect_document_grounding():
    assert user_expects_document_grounding(
        "What is this playbook mainly about, in plain language?"
    )
    assert user_expects_document_grounding("What was Acme Corp Q3 revenue in the finance flash?")
    assert user_expects_document_grounding(
        "What does the playbook say about p99 latency requirements?"
    )
    assert not wants_no_retrieval_fastpath(
        "What is this playbook mainly about, in plain language?"
    )
