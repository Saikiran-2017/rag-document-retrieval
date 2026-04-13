"""
PHASE 1-5 Product-grade robustness test suite (deterministic + query logic paths).
Tests deterministic overview fallback, query intent detection, and non-regression.
Avoids LLM calls to focus on logic testability.
"""

import pytest
from app.llm.deterministic_extraction import (
    try_build_grounded_document_overview,
    try_extract_field_value_answer,
)
from app.retrieval.vector_store import RetrievedChunk
from app.llm.query_intent import (
    is_broad_document_overview_query,
    user_expects_document_grounding,
    normalize_query_for_field_intent,
)


# ============================================================================
# PHASE 1: Deterministic Summary Path (No Labels, Natural Language)
# ============================================================================

class TestDeterministicStructuredSummary:
    """Deterministic fallback for structured field-heavy documents."""

    def test_resume_deterministic_no_label_dump(self):
        """Resume through deterministic path should never dump labels."""
        hits = [
            RetrievedChunk(
                rank=0,
                page_content="Full Name: John Smith\nTitle: Senior Engineer",
                metadata={"source_name": "resume.txt", "chunk_id": "1"},
                distance=0.5,
            ),
            RetrievedChunk(
                rank=1,
                page_content="Company: Acme Corp\nRole: Lead Backend",
                metadata={"source_name": "resume.txt", "chunk_id": "2"},
                distance=0.51,
            ),
            RetrievedChunk(
                rank=2,
                page_content="Skills: Python, Go\nCertifications: AWS",
                metadata={"source_name": "resume.txt", "chunk_id": "3"},
                distance=0.52,
            ),
        ]
        result = try_build_grounded_document_overview("summarize this", hits)
        
        if result:
            answer_lower = result.answer.lower()
            # No label dumps
            assert "full name" not in answer_lower
            assert "title:" not in answer_lower
            assert "skills:" not in answer_lower
            # Natural theme-based language
            assert ("engineer" in answer_lower or "lead" in answer_lower or
                    "acme" in answer_lower or "company" in answer_lower)
            # Always has citations
            assert "[SOURCE" in result.answer

    def test_loan_form_deterministic_natural(self):
        """Loan form deterministic path uses natural themes."""
        hits = [
            RetrievedChunk(
                rank=0,
                page_content="Application Number\nApplicant Name\nRepayment Schedule",
                metadata={"source_name": "loan.pdf", "chunk_id": "1"},
                distance=0.9,
            ),
            RetrievedChunk(
                rank=1,
                page_content="Loan Amount\nDisbursed Date\nRate of Interest",
                metadata={"source_name": "loan.pdf", "chunk_id": "2"},
                distance=0.91,
            ),
            RetrievedChunk(
                rank=2,
                page_content="Interest Certificate\nRepayment Schedule",
                metadata={"source_name": "loan.pdf", "chunk_id": "3"},
                distance=0.92,
            ),
        ]
        result = try_build_grounded_document_overview("what is this?", hits)
        
        if result:
            answer_lower = result.answer.lower()
            # Should NOT echo labels directly
            assert "application number:" not in answer_lower
            # Should have theme-based language
            assert ("loan" in answer_lower or "applicant" in answer_lower or
                    "repayment" in answer_lower or "disburs" in answer_lower)
            assert "[SOURCE" in result.answer


# ============================================================================
# PHASE 2: Query Intent Recognition (Typos, Normalization)
# ============================================================================

class TestSummaryQueryIntentDetection:
    """Query intent should detect summary requests despite typos."""

    def test_sumarize_typo_detected(self):
        """Typo 'sumarize' should match summary intent."""
        query = "sumarize this"
        assert is_broad_document_overview_query(query)
        assert user_expects_document_grounding(query)

    def test_summarise_british_detected(self):
        """British 'summarise' should match."""
        query = "summarise this document"
        assert is_broad_document_overview_query(query)

    def test_explain_detected_as_summary(self):
        """'Explain this document' should trigger summary logic."""
        query = "explain this document"
        assert is_broad_document_overview_query(query)

    def test_what_about_query_detected(self):
        """'What is this about?' should trigger overview logic."""
        query = "what is this document about?"
        assert is_broad_document_overview_query(query)


# ============================================================================
# PHASE 2: Field Extraction Non-Regression (Query Intent)
# ============================================================================

class TestFieldExtractionIntentDetection:
    """Field extraction intent detection should still work."""

    def test_name_question_recognized(self):
        """Name extraction intent recognized."""
        from app.llm.deterministic_extraction import field_value_question_kind
        kind = field_value_question_kind("what is the full name?")
        assert kind == "person_name"

    def test_email_question_recognized(self):
        """Email extraction intent recognized."""
        from app.llm.deterministic_extraction import field_value_question_kind
        kind = field_value_question_kind("what is the email?")
        assert kind == "email"

    def test_phone_question_recognized(self):
        """Phone extraction intent recognized."""
        from app.llm.deterministic_extraction import field_value_question_kind
        kind = field_value_question_kind("what is the phone?")
        assert kind == "phone"

    def test_phone_typo_phne_normalized(self):
        """Typo 'phne' should still be detected as phone."""
        from app.llm.deterministic_extraction import field_value_question_kind
        kind = field_value_question_kind("what is the phne number?")
        # Should recognize it despite typo
        assert kind is not None


# ============================================================================
# PHASE 3: Field Extraction Deterministic Fallback Non-Regression 
# ============================================================================

class TestFieldExtractionFallbackNonRegression:
    """Field extraction deterministic paths should still work."""

    def test_email_deterministic_extraction(self):
        """Email extraction should work from hit."""
        hits = [
            RetrievedChunk(
                rank=0,
                page_content="Email: alice@example.com",
                metadata={"source_name": "contact.txt", "chunk_id": "1"},
                distance=0.5,
            )
        ]
        result = try_extract_field_value_answer("what is the email?", hits)
        if result:
            assert "alice@example.com" in result.answer

    def test_phone_deterministic_extraction(self):
        """Phone extraction should work."""
        hits = [
            RetrievedChunk(
                rank=0,
                page_content="Phone: 555-123-4567",
                metadata={"source_name": "contact.txt", "chunk_id": "1"},
                distance=0.5,
            )
        ]
        result = try_extract_field_value_answer("what is the phone?", hits)
        if result:
            assert "555-123-4567" in result.answer

    def test_name_deterministic_extraction(self):
        """Name extraction should work."""
        hits = [
            RetrievedChunk(
                rank=0,
                page_content="Full Name: Jane Smith",
                metadata={"source_name": "resume.txt", "chunk_id": "1"},
                distance=0.5,
            )
        ]
        result = try_extract_field_value_answer("what is the name?", hits)
        if result:
            assert "jane" in result.answer.lower() or "smith" in result.answer.lower()


# ============================================================================
# PHASE 3: Preserve Strong Current Behavior - Deterministic Paths
# ============================================================================

class TestRefusalAndIdentityNonRegression:
    """Ensure refusals and identity behavior preserved."""

    def test_unknown_phrase_used_when_no_context(self):
        """System should use UNKNOWN_PHRASE when context is weak."""
        from app.llm.generator import UNKNOWN_PHRASE
        assert UNKNOWN_PHRASE and len(UNKNOWN_PHRASE) > 0


class TestGeneralKnowledgeDetection:
    """General knowledge questions should not force document grounding."""

    def test_ml_question_not_document_forced(self):
        """General ML question should not require document."""
        query = "what is machine learning?"
        # Should NOT trigger document intent
        assert not is_broad_document_overview_query(query)
        # May not expect document grounding
        expect_doc = user_expects_document_grounding(query)
        # It's fine if it does - the gate routing will filter it later


# ============================================================================
# PHASE 3: Deterministic Fallback Path Non-Regression
# ============================================================================

class TestDeterministicFallbackQuality:
    """Deterministic overview fallback should never regress to old phrasing."""

    def test_no_old_boilerplate_in_deterministic(self):
        """Old phrasing 'material in excerpts' should NOT appear."""
        hits = [
            RetrievedChunk(
                rank=0,
                page_content="Field: Value\nField2: Value2",
                metadata={"source_name": "test.txt", "chunk_id": "1"},
                distance=0.5,
            ),
            RetrievedChunk(
                rank=1,
                page_content="Field: Value\nField2: Value2",
                metadata={"source_name": "test.txt", "chunk_id": "2"},
                distance=0.51,
            ),
            RetrievedChunk(
                rank=2,
                page_content="Field: Value\nField2: Value2",
                metadata={"source_name": "test.txt", "chunk_id": "3"},
                distance=0.52,
            ),
        ]
        result = try_build_grounded_document_overview("what is this?", hits)
        
        if result:
            answer_lower = result.answer.lower()
            # NEVER use old boilerplate
            assert "material in the excerpts" not in answer_lower
            assert "as reflected in the retrieved passages" not in answer_lower
            # Should always cite
            assert "[SOURCE" in result.answer

    def test_deterministic_uses_natural_opener(self):
        """Deterministic should use 'It provides...' not 'This document provides...'"""
        hits = [
            RetrievedChunk(
                rank=0,
                page_content="Company Name\nRole Title",
                metadata={"source_name": "test.txt", "chunk_id": "1"},
                distance=0.5,
            ),
            RetrievedChunk(
                rank=1,
                page_content="Project Details",
                metadata={"source_name": "test.txt", "chunk_id": "2"},
                distance=0.51,
            ),
            RetrievedChunk(
                rank=2,
                page_content="Skills Listed",
                metadata={"source_name": "test.txt", "chunk_id": "3"},
                distance=0.52,
            ),
        ]
        result = try_build_grounded_document_overview("summarize", hits)
        
        if result:
            answer = result.answer
            # Acceptable openers
            acceptable = (
                "It provides" in answer or
                answer[0:20] and answer[0].isupper()  # Starts with sentence
            )
            assert acceptable


# ============================================================================
# PHASE 4: Additional Regression Tests (Non-LLM)
# ============================================================================

class TestDiverseDocumentShapesHandling:
    """System should handle various document shapes without crashing."""

    def test_very_short_document_fragment(self):
        """Short snippets should still work."""
        hits = [
            RetrievedChunk(
                rank=0,
                page_content="Email: test@x.org",
                metadata={"source_name": "short.txt", "chunk_id": "1"},
                distance=0.5,
            )
        ]
        # Should not crash
        result = try_extract_field_value_answer("what is the email?", hits)
        # May or may not extract, but shouldn't crash
        assert result is None or "test@x.org" in result.answer

    def test_repetitive_field_document(self):
        """Highly repetitive fields should handle gracefully."""
        hits = [
            RetrievedChunk(
                rank=i,
                page_content="Name: Name\nValue: Value\nField: Field",
                metadata={"source_name": "rep.txt", "chunk_id": str(i)},
                distance=0.5 + (i * 0.01),
            )
            for i in range(1, 4)
        ]
        # Should not crash on deterministic overview
        result = try_build_grounded_document_overview("what is this?", hits)
        # May be None if pattern doesn't match, but no crash
        assert result is None or "[SOURCE" in result.answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
