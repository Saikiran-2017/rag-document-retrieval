from __future__ import annotations

from app.llm.deterministic_extraction import (
    try_build_grounded_document_overview,
    try_extract_field_value_answer,
)
from app.retrieval.vector_store import RetrievedChunk


def _hit(text: str, *, source_name: str = "sample_loan_record_mock.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        rank=0,
        page_content=text,
        metadata={"source_name": source_name, "chunk_id": "c1", "page_number": 1, "file_path": "data/raw/x.pdf"},
        distance=0.9,
    )


def test_credila_doc_about_deterministic_summary_when_structure_is_obvious() -> None:
    hits = [
        _hit("Process Loan - Loan Details\nApplicant Name\nRepayment Schedule\nInterest Certificate"),
        _hit("Loan Disbursed Amount\nLoan Disbursed Date\nRate of Interest"),
        _hit("Application Number\nApplicant Name\nLoan Disbursed Amount"),
        _hit("Some other heading-like line\nRepayment Schedule"),
    ]
    out = try_build_grounded_document_overview("what is this document about?", hits)
    assert out is not None
    low = out.answer.lower()
    assert "[source" in low
    assert "structured" in low or "administrative" in low or "loan" in low


def test_credila_how_much_is_my_loan_extracts_amount() -> None:
    hits = [
        _hit("Applicant Name: HOLDER ALPHA"),
        _hit("Loan Disbursed Amount 1,026,904.00\nRepayment Schedule: Monthly"),
    ]
    out = try_extract_field_value_answer("how much is my loan?", hits)
    assert out is not None
    assert "1,026,904.00" in out.answer
    assert "[SOURCE 2]" in out.answer


def test_credila_applicant_name_extracts_name() -> None:
    hits = [
        _hit("Loan Details\nApplicant Name: HOLDER ALPHA\nApplication Number: ABCD1234"),
    ]
    out = try_extract_field_value_answer("what is the applicant name?", hits)
    assert out is not None
    assert "HOLDER ALPHA" in out.answer
    assert "[SOURCE 1]" in out.answer


def test_credila_application_number_extracts_id() -> None:
    hits = [
        _hit("Application No: ABCD-12345\nLoan Disbursed Amount: 1,026,904.00"),
    ]
    out = try_extract_field_value_answer("what is the application number?", hits)
    assert out is not None
    assert "ABCD-12345" in out.answer
    assert "[SOURCE 1]" in out.answer


def test_negative_case_no_field_value_match_returns_none() -> None:
    hits = [
        _hit("This excerpt does not contain the requested field."),
        _hit("Still nothing about applicant name or application number."),
    ]
    assert try_extract_field_value_answer("what is the applicant name?", hits) is None

