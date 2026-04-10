"""
Transcript-style regression: multi-turn behavior without live LLM or real uploads.

Validates that conversation history shapes retrieval hints the same way a UI transcript would.
"""

from __future__ import annotations

from app.llm.conversation_context import build_conversation_retrieval_hints


def test_transcript_style_two_turn_follow_up() -> None:
    transcript = [
        {"role": "user", "content": "Review the fixture record for case REF-UNIT-7."},
        {
            "role": "assistant",
            "content": "It includes labeled rows for subject and facility [SOURCE 1].",
            "mode": "grounded",
            "sources": [
                {"source_name": "fixture_record_unit7.txt", "source_number": 1},
            ],
        },
    ]
    h = build_conversation_retrieval_hints("what is the file name?", transcript)
    assert h.force_document_scoped_routing
    assert h.focus_source_name == "fixture_record_unit7.txt"

    h2 = build_conversation_retrieval_hints("did they include any address?", transcript)
    assert h2.force_document_scoped_routing
    assert h2.relax_lookup_gate is True
