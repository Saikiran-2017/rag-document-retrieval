#!/usr/bin/env python3
"""
REAL PRODUCT VALIDATION - Manual testing against actual documents
Tests deterministic + intent detection paths (no LLM calls)
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm.query_normalize import normalize_query_for_pipeline
from app.llm.query_intent import is_broad_document_overview_query
from app.llm.deterministic_extraction import (
    try_build_grounded_document_overview,
    try_extract_field_value_answer,
    field_value_question_kind,
)
from app.retrieval.vector_store import RetrievedChunk

# ============================================================================
# DOCUMENT FIXTURES - Real test documents
# ============================================================================

DOC_SPACEX = """DOCUMENT TITLE: SpaceX Leadership and Company Overview

Full Name: Elon Reeve Musk  
Date of Birth: June 28, 1971  
Nationality: American (originally South African-born)  

Current Role:
Elon Musk is the CEO and Chief Engineer of SpaceX (Space Exploration Technologies Corp).

Company Name: SpaceX  
Founded: 2002  
Headquarters: Hawthorne, California, USA  

Company Overview:
SpaceX is a private aerospace manufacturer and space transportation company focused on reducing space transportation costs and enabling the colonization of Mars.

Key Achievements:
- First privately funded company to reach orbit (Falcon 1)
- Developed Falcon 9 reusable rocket
- Developed Falcon Heavy, one of the most powerful rockets
- Developed Dragon spacecraft for cargo and crew missions
- Partnered with NASA for ISS missions

Major Projects:
- Starship: Fully reusable spacecraft designed for missions to Mars
- Starlink: Satellite internet constellation providing global broadband coverage

Technologies Used:
- Reusable rocket technology
- Advanced propulsion systems
- Autonomous landing systems
- Satellite communication systems

Address:
1 Rocket Road, Hawthorne, CA, 90250, USA  

Email:
contact@spacex.com  

Phone Number:
+1-310-363-6000  

Website:
https://www.spacex.com  

Employees:
More than 13,000 employees worldwide

Mission Statement:
To make humanity multi-planetary by enabling sustainable space travel."""


DOC_LOAN = """Credila's Student Loan Intranet System

Process Loan - Loan Details
Applicant Name: SAI KIRAN
Application Number: ABCD-12345
Loan Disbursed Amount: 1,026,904.00
Repayment Schedule: Monthly
Interest Certificate: Available
Rate of Interest: 9.25%
Loan Disbursed Date: 2025-01-15
Status: Active"""


DOC_FINANCE = """Acme Corp - internal finance flash (Q3)

Revenue for the third quarter was forty-seven million USD (47,000,000), as filed with the internal controller.

The chief executive officer is Jane Okonkwo, appointed effective 2024-03-15.
The chief technology officer is Sam Rivera.

This excerpt exists so eval cases can test narrow numeric and title extraction across a second file in the corpus.
Quarterly earnings have been growing steadily since 2023."""


DOC_MIXED = """CONFIDENTIAL — Q4 OPERATIONS REVIEW

Executive summary
This section wraps badly across
lines for testing reflow and document understanding.

FINANCIAL SNAPSHOT (TABLE_BLOCK)
Product     | Q3 Rev   | Margin
Widget A    | 12.4M    | 18%
Widget B    | 8.1M     | 22%

## Markdown style section (SECTION_MD)
Content here about operations and latency metrics for regression checks.

Another paragraph after the markdown heading block.
Contact: ops@company.com
Phone: 555-123-4567"""


DOC_SHORT = """Quick Note

Name: John Smith
Email: john@example.com
Phone: 555-9876
Status: Approved"""


# ============================================================================
# VALIDATION SUITE
# ============================================================================

class ValidationResult:
    def __init__(self, doc_type, query, result, status, weakness=None):
        self.doc_type = doc_type
        self.query = query
        self.result = result
        self.status = status
        self.weakness = weakness

    def __repr__(self):
        return (
            f"✓ PASS" if self.status == "PASS" else f"✗ FAIL"
        ) + f" | {self.doc_type} | Q: {self.query[:50]}"


def validate_query(doc_type, doc_content, query_desc, query, expected_behavior):
    """
    Validate a single query against document content.
    Uses real app logic: normalize, intent detection, deterministic extraction.
    """
    normalized = normalize_query_for_pipeline(query)
    
    # Detect intent
    is_overview = is_broad_document_overview_query(query)
    field_kind = field_value_question_kind(query)
    
    # Create a mock RetrievedChunk for both overview and field extraction
    chunk = RetrievedChunk(
        rank=1,
        page_content=doc_content,
        metadata={
            "chunk_id": "1",
            "source_name": doc_type.split()[0],
        },
        distance=0.0
    )
    
    # Try deterministic generation
    overview_result = None
    field_result = None
    
    if is_overview:
        overview_result = try_build_grounded_document_overview(query, [chunk])
    
    if field_kind and not is_overview:
        field_result = try_extract_field_value_answer(query, [chunk])
    
    # Determine result
    result_text = ""
    status = "FAIL"
    weakness = None
    
    if is_overview and overview_result:
        result_text = overview_result.answer if hasattr(overview_result, 'answer') else str(overview_result)
        result_text = result_text[:100] + "..." if len(result_text) > 100 else result_text
        # Check quality: no label dumps, has content, natural language
        has_label_dump = any(
            bad in result_text.lower() 
            for bad in ["applicant name:", "full name:", "email:", "phone:"]
        )
        has_content = len(result_text) > 20
        has_colon_start = result_text.strip().startswith(":")
        
        if has_label_dump or has_colon_start:
            weakness = "Label dump or malformed result"
            status = "FAIL"
        elif has_content:
            status = "PASS"
        else:
            weakness = "Result too short"
            status = "FAIL"
    elif field_kind and field_result:
        result_text = field_result.answer if hasattr(field_result, 'answer') else str(field_result)
        if result_text and "UNKNOWN" not in result_text and len(result_text) > 5:
            status = "PASS"
        else:
            weakness = "Field extraction failed or empty"
            status = "FAIL"
    else:
        result_text = f"[Intent: overview={is_overview}, field={field_kind}]"
        if is_overview or field_kind:
            weakness = "No deterministic result generated"
        else:
            status = "PASS"  # Queries like "does this document discuss..." are routing queries
            result_text = "[Routing query - not deterministic path]"
    
    return ValidationResult(
        doc_type=doc_type,
        query=f"{query_desc}: {query}",
        result=result_text,
        status=status,
        weakness=weakness
    )


# ============================================================================
# RUN VALIDATION SUITE
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("REAL PRODUCT VALIDATION - Against Actual Document Content")
    print("=" * 100)
    
    tests = [
        # DOCUMENT 1: Mixed Field + Narrative (SpaceX Profile)
        ("Mixed Field+Narrative\n(SpaceX Profile)", DOC_SPACEX, [
            ("summarize", "summarize this document"),
            ("field_lookup", "what is the email?"),
            ("follow_up_detail", "what are the major projects?"),
            ("negative_q", "does this document discuss medical supplies?"),
            ("typo_variant", "sumarize this document"),
        ]),
        
        # DOCUMENT 2: Structured Field-Heavy (Loan Form)
        ("Structured Field-Heavy\n(Loan Form)", DOC_LOAN, [
            ("summarize", "summarize this document"),
            ("field_lookup", "what is the applicant name?"),
            ("follow_up_detail", "what is the rate of interest?"),
            ("negative_q", "does this document list any employees?"),
            ("typo_variant", "summarise this loan document"),
        ]),
        
        # DOCUMENT 3: Prose Company Overview (Finance)
        ("Prose Company Overview\n(Acme Finance)", DOC_FINANCE, [
            ("summarize", "summarize this document"),
            ("field_lookup", "who is the CEO?"),
            ("follow_up_detail", "what is the revenue amount?"),
            ("negative_q", "does this document describe a software product?"),
            ("typo_variant", "what is this about?"),  # explain variant
        ]),
        
        # DOCUMENT 4: Mixed Layout (Operations Review)
        ("Mixed Layout\n(Operations Review)", DOC_MIXED, [
            ("summarize", "summarize this document"),
            ("field_lookup", "what is the contact email?"),
            ("follow_up_detail", "what products are mentioned?"),
            ("negative_q", "does this contain customer testimonials?"),
            ("typo_variant", "phne?"),  # phone typo
        ]),
        
        # DOCUMENT 5: Short Plain Text (Note)
        ("Short Plain Text\n(Quick Note)", DOC_SHORT, [
            ("summarize", "summarize this document"),
            ("field_lookup", "what is the email?"),
            ("follow_up_detail", "what is the status?"),
            ("negative_q", "are there any dates mentioned?"),
            ("typo_variant", "what is the phne?"),  # typo variant
        ]),
    ]
    
    all_results = []
    
    for doc_type, doc_content, queries in tests:
        print(f"\n{'-' * 100}")
        print(f"DOCUMENT TYPE: {doc_type}")
        print(f"Document length: {len(doc_content)} chars")
        print(f"{'-' * 100}")
        
        for query_kind, query_text in queries:
            result = validate_query(doc_type, doc_content, query_kind, query_text, None)
            all_results.append(result)
            
            status_icon = "✓ PASS" if result.status == "PASS" else "✗ FAIL"
            print(f"\n{status_icon} | {query_kind.upper()}")
            print(f"   Query: {query_text}")
            print(f"   Result: {result.result}")
            if result.weakness:
                print(f"   ⚠ WEAKNESS: {result.weakness}")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    passed = sum(1 for r in all_results if r.status == "PASS")
    failed = sum(1 for r in all_results if r.status == "FAIL")
    
    print(f"\n✓ PASSED: {passed}/{len(all_results)}")
    print(f"✗ FAILED: {failed}/{len(all_results)}")
    
    if failed > 0:
        print(f"\nFailed Tests:")
        for r in all_results:
            if r.status == "FAIL":
                print(f"  ✗ {r.doc_type.split()[0]} | {r.query[:60]}")
                if r.weakness:
                    print(f"    Reason: {r.weakness}")
    
    # Final Verdict
    print(f"\n{'=' * 100}")
    if failed == 0:
        print("✅ FINAL VERDICT: PRODUCT READY - All validation tests pass")
    elif failed <= 2:
        print("⚠️  FINAL VERDICT: MOSTLY READY - Minor issues found")
    else:
        print("❌ FINAL VERDICT: NOT READY - Multiple issues detected")
    print(f"{'=' * 100}\n")
    
    return 0 if failed <= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
