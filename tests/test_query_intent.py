"""Document-scope heuristics: avoid false general fast paths on doc-shaped questions."""

from __future__ import annotations

from app.llm.query_intent import is_broad_document_overview_query, user_expects_document_grounding
from app.services.chat_service import wants_no_retrieval_fastpath


def test_main_points_triggers_retrieval_not_fastpath():
    assert user_expects_document_grounding("What are the main points?")
    assert not wants_no_retrieval_fastpath("What are the main points?")


def test_main_topics_triggers_retrieval():
    assert user_expects_document_grounding("What are the main topics in this file?")
    assert not wants_no_retrieval_fastpath("What are the main topics in this file?")


def test_casual_greeting_can_fastpath():
    assert wants_no_retrieval_fastpath("Hello there")


def test_broad_overview_flag():
    assert is_broad_document_overview_query("What is this document about?")
    assert is_broad_document_overview_query("Summarize this file")
    assert not is_broad_document_overview_query("What is the revenue in Q3 per the table?")
