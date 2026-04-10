"""Deterministic metadata + extra field patterns (privacy-safe synthetic text only)."""

from __future__ import annotations

from app.llm.deterministic_extraction import try_answer_document_metadata_question, try_extract_field_value_answer
from app.retrieval.vector_store import RetrievedChunk


def _hit(text: str, *, source: str = "sample_record_a.txt") -> RetrievedChunk:
    return RetrievedChunk(
        rank=0,
        page_content=text,
        metadata={"source_name": source, "chunk_id": "c1", "page_number": 1, "file_path": "data/raw/sample_record_a.txt"},
        distance=0.9,
    )


def test_metadata_filename_from_hits() -> None:
    hits = [_hit("Subject Name: HOLDER BETA\nEmail: ops@example.invalid")]
    out = try_answer_document_metadata_question("what is the document name?", hits)
    assert out is not None
    assert "sample_record_a.txt" in out.answer
    assert "[SOURCE 1]" in out.answer


def test_extract_email_phone_address() -> None:
    hits = [
        _hit(
            "Email: contact@example.invalid\n"
            "Contact Number: +1 (555) 010-0199\n"
            "Current Address: 100 Main Street, Suite 2, Example City, EX 12345"
        ),
    ]
    e = try_extract_field_value_answer("what is the email on file?", hits)
    assert e is not None
    assert "example.invalid" in e.answer
    p = try_extract_field_value_answer("what is the contact number?", hits)
    assert p is not None
    assert "555" in p.answer
    a = try_extract_field_value_answer("what is the current address?", hits)
    assert a is not None
    assert "Main Street" in a.answer


def test_his_name_maps_to_labeled_person_field() -> None:
    hits = [_hit("Applicant Name: HOLDER GAMMA")]
    out = try_extract_field_value_answer("what is his name?", hits)
    assert out is not None
    assert "HOLDER GAMMA" in out.answer


def test_negative_missing_field_returns_none() -> None:
    hits = [_hit("Only unrelated boilerplate text here.")]
    assert try_extract_field_value_answer("what is the passport number?", hits) is None
