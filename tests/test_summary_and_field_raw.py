"""Summary intent equivalence and raw-library field extraction."""

from __future__ import annotations

from pathlib import Path

from app.llm.deterministic_extraction import (
    field_value_question_kind,
    try_build_grounded_document_overview,
    try_extract_field_from_raw_library,
    try_extract_field_value_answer,
)
from app.llm.query_intent import is_broad_document_overview_query, user_expects_document_grounding
from app.retrieval.vector_store import RetrievedChunk
from app.services.chat_service import wants_no_retrieval_fastpath


def test_summary_intent_variants_expect_retrieval() -> None:
    for q in (
        "summarize this document",
        "give summary",
        "give me a summary",
        "explain this document",
        "summarize",
    ):
        assert is_broad_document_overview_query(q), q
        assert user_expects_document_grounding(q), q
        assert not wants_no_retrieval_fastpath(q), q


def test_field_questions_detect_kind() -> None:
    assert field_value_question_kind("what is the full name") == "person_name"
    assert field_value_question_kind("what is the email") == "email"
    assert field_value_question_kind("what is the contact email") == "email"
    assert field_value_question_kind("what is the phone number") == "phone"
    assert field_value_question_kind("what is phne number") == "phone"
    assert field_value_question_kind("what is the website") == "website"


def test_raw_library_extracts_email_when_chunk_missing(tmp_path: Path) -> None:
    f = tmp_path / "contact.txt"
    f.write_text("Random intro line\nContact Email: found@example.org\nFooter\n", encoding="utf-8")
    rb = try_extract_field_from_raw_library("what is the email?", [f])
    assert rb is not None
    ext, hits = rb
    assert "found@example.org" in ext.answer
    assert hits and hits[0].metadata.get("source_name") == "contact.txt"


def test_raw_library_email_any_fallback(tmp_path: Path) -> None:
    f = tmp_path / "plain.txt"
    f.write_text("No label at all but reach us at plain@space.test today.\n", encoding="utf-8")
    rb = try_extract_field_from_raw_library("what is the email?", [f])
    assert rb is not None
    assert "plain@space.test" in rb[0].answer


def test_prioritize_structured_field_orders_email_chunk() -> None:
    from app.retrieval.context_selection import prioritize_structured_field_hits

    h1 = RetrievedChunk(
        rank=0,
        page_content="Some unrelated narrative without fields.",
        metadata={"source_name": "a.txt"},
        distance=0.5,
    )
    h2 = RetrievedChunk(
        rank=1,
        page_content="Email: x@y.com\nPhone: 555-000-1111",
        metadata={"source_name": "a.txt"},
        distance=0.6,
    )
    out = prioritize_structured_field_hits([h1, h2], "what is the email")
    assert (out[0].page_content or "").startswith("Email")


def test_overview_on_summarize_query_with_hits() -> None:
    hits = [
        RetrievedChunk(
            rank=0,
            page_content="Full Name: A B\nEmail: a@b.co\nPhone: 555-111-2222",
            metadata={"source_name": "one.txt", "chunk_id": "1"},
            distance=0.4,
        ),
        RetrievedChunk(
            rank=1,
            page_content="Website: https://example.com\nAddress: 1 Main St",
            metadata={"source_name": "one.txt", "chunk_id": "2"},
            distance=0.41,
        ),
        RetrievedChunk(
            rank=2,
            page_content="Company: Acme\nRole: Engineer",
            metadata={"source_name": "one.txt", "chunk_id": "3"},
            distance=0.42,
        ),
    ]
    o = try_build_grounded_document_overview("summarize this document", hits)
    assert o is not None and "[source" in o.answer.lower()
    assert "provides an overview" in o.answer.lower()


def test_chunk_hit_extract_phone() -> None:
    h = [
        RetrievedChunk(
            rank=0,
            page_content="Phone: +1 (800) 555-0199",
            metadata={"source_name": "p.txt"},
            distance=0.3,
        )
    ]
    ex = try_extract_field_value_answer("what is the phone number?", h)
    assert ex is not None and "555" in ex.answer
