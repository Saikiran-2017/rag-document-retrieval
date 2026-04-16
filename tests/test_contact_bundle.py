"""Grouped contact bundle extraction for continuation queries like "details" or "contact info"."""

from __future__ import annotations

from app.llm.deterministic_extraction import (
    try_extract_contact_info_bundle_answer,
    try_extract_field_value_answer,
)
from app.retrieval.vector_store import RetrievedChunk


def _make_chunk(content: str, source_name: str = "test.txt", source_number: int = 1) -> RetrievedChunk:
    """Helper to create a test chunk."""
    return RetrievedChunk(
        rank=0,
        page_content=content,
        metadata={"source_name": source_name, "chunk_id": f"chunk-{source_number}"},
        distance=0.0,
    )


def test_bundle_requires_contact_query() -> None:
    """Bundle extraction only triggers when query explicitly asks for contact/details."""
    chunk = _make_chunk("Email: test@company.com\nPhone: (555) 123-4567")
    
    # Generic queries should return None
    result = try_extract_contact_info_bundle_answer("what is this document about", [chunk])
    assert result is None
    
    # Explicit "details" or "contact" keywords should trigger
    result = try_extract_contact_info_bundle_answer("details", [chunk])
    assert result is not None
    
    result2 = try_extract_contact_info_bundle_answer("contact info", [chunk])
    assert result2 is not None


def test_bundle_requires_at_least_two_fields() -> None:
    """Bundle only returns when at least 2 contact fields are found."""
    # Only email
    chunk_email_only = _make_chunk("Email: contact@example.com")
    result = try_extract_contact_info_bundle_answer("contact details", [chunk_email_only])
    assert result is None
    
    # Email + phone - should return
    chunk_email_phone = _make_chunk("Email: contact@example.com\nPhone: (555) 123-4567")
    result = try_extract_contact_info_bundle_answer("contact details", [chunk_email_phone])
    assert result is not None
    assert "Email" in result.answer
    assert "Phone" in result.answer


def test_bundle_after_email_query() -> None:
    """'details' after email lookup should return grouped contact block."""
    chunk = _make_chunk(
        "Applicant Name: John Doe\nEmail: john@example.com\nPhone: (555) 123-4567\nAddress: 123 Main St, City, ST 12345"
    )
    
    # First ask for email
    email_result = try_extract_field_value_answer("what is the email", [chunk])
    assert email_result is not None
    assert "john@example.com" in email_result.answer
    
    # Then ask for details -> should get bundle
    bundle = try_extract_contact_info_bundle_answer("details", [chunk])
    assert bundle is not None
    assert "Email" in bundle.answer
    assert "Phone" in bundle.answer
    assert "john@example.com" in bundle.answer
    assert "555" in bundle.answer  # Phone has 555


def test_bundle_citations_preserved() -> None:
    """Each field in bundle should have [SOURCE n] citation."""
    chunk = _make_chunk(
        "Email: info@company.com\nPhone: (555) 123-4567\nWebsite: www.company.com",
        source_name="company.txt",
    )
    
    result = try_extract_contact_info_bundle_answer("contact info", [chunk])
    assert result is not None
    
    # Check all citations are present
    assert "[SOURCE 1]" in result.answer
    # All sources should be same (source 1)
    assert result.used_source_numbers == (1,)


def test_bundle_multiple_sources() -> None:
    """Bundle can collect fields from multiple source chunks."""
    chunk1 = _make_chunk("Email: support@company.com", source_number=1)
    chunk2 = _make_chunk("Phone: +1-800-555-1234", source_number=2)
    chunk3 = _make_chunk("Address: 100 Corporate Drive, Suite 500, USA", source_number=3)
    
    result = try_extract_contact_info_bundle_answer("contact details", [chunk1, chunk2, chunk3])
    assert result is not None
    
    # Should have fields from all sources
    assert "Email" in result.answer
    assert "Phone" in result.answer
    assert "Address" in result.answer
    assert "[SOURCE 1]" in result.answer
    assert "[SOURCE 2]" in result.answer
    assert "[SOURCE 3]" in result.answer
    assert result.used_source_numbers == (1, 2, 3)


def test_bundle_email_validation() -> None:
    """Email extraction should be strict."""
    # Valid email with phone
    chunk_valid = _make_chunk("Email: user.name+tag@example.co.uk\nPhone: (555) 123-4567")
    result = try_extract_contact_info_bundle_answer("contact info", [chunk_valid])
    assert result is not None
    assert "user.name+tag@example.co.uk" in result.answer
    
    # Invalid email should be ignored, so only website + phone if available
    chunk_bad_email = _make_chunk("Email: invalid.email@\nPhone: (555) 123-4567\nWebsite: www.example.com")
    result2 = try_extract_contact_info_bundle_answer("contact details", [chunk_bad_email])
    assert result2 is not None
    # Should have phone + website (email invalid)
    assert "Phone" in result2.answer
    assert "Website" in result2.answer
    assert "[SOURCE 1]" in result2.answer


def test_bundle_phone_validation() -> None:
    """Phone should be validated properly."""
    # Valid: 10+ digits
    chunk_valid = _make_chunk("Phone: (555) 123-4567\nEmail: test@example.com")
    result = try_extract_contact_info_bundle_answer("contact", [chunk_valid])
    assert result is not None
    assert "Phone" in result.answer
    assert "555" in result.answer


def test_bundle_website_extraction() -> None:
    """Website should be extracted in various formats."""
    chunk = _make_chunk("Website: https://www.example.com\nEmail: info@example.com")
    result = try_extract_contact_info_bundle_answer("details", [chunk])
    assert result is not None
    assert "Website" in result.answer
    assert "https://www.example.com" in result.answer


def test_bundle_address_extraction() -> None:
    """Address should include location markers and be reasonable length."""
    chunk = _make_chunk(
        "Address: 456 Market Street, Suite 200, San Francisco, CA 94102\nPhone: (415) 555-9876\nEmail: mail@example.com"
    )
    result = try_extract_contact_info_bundle_answer("contact info", [chunk])
    assert result is not None
    assert "Address" in result.answer
    assert "San Francisco" in result.answer
    assert "456 Market Street" in result.answer


def test_bundle_query_keywords() -> None:
    """Bundle should trigger on various detail-related keywords."""
    chunk = _make_chunk("Email: test@example.com\nPhone: (555) 123-4567")
    
    # Only test keywords that clearly indicate contact details
    for query in ["details", "contact info", "contact details", "more information", "what else"]:
        result = try_extract_contact_info_bundle_answer(query, [chunk])
        assert result is not None, f"Failed for query: {query}"


def test_bundle_concise_output() -> None:
    """Bundle output should be concise and formatted clearly."""
    chunk = _make_chunk("Email: contact@example.com\nPhone: +1-800-555-0123")
    result = try_extract_contact_info_bundle_answer("contact details", [chunk])
    assert result is not None
    
    # Check format is clean
    assert "Here is the contact information" in result.answer
    assert "**Email**:" in result.answer
    assert "**Phone**:" in result.answer
    # No hallucination (website not mentioned if not present)
    assert "**Website**:" not in result.answer


def test_bundle_no_hallucination() -> None:
    """Bundle should never invent missing fields."""
    # Only email - not enough for bundle
    chunk = _make_chunk("Email: test@example.com")
    result = try_extract_contact_info_bundle_answer("can you give me all contact details", [chunk])
    assert result is None
    
    # With phone, no website mentioned unless it exists
    chunk2 = _make_chunk("Email: test@example.com\nPhone: (555) 123-4567")
    result2 = try_extract_contact_info_bundle_answer("all contact info", [chunk2])
    assert result2 is not None
    assert "**Website**:" not in result2.answer
    assert "**Address**:" not in result2.answer
    # Should only have email and phone
    assert "**Email**:" in result2.answer
    assert "**Phone**:" in result2.answer
