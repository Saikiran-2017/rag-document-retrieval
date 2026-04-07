"""Round-robin chunk selection for multi-source broad QA."""

from __future__ import annotations

from app.retrieval.context_selection import diversify_chunks_round_robin, select_generation_context
from app.retrieval.vector_store import RetrievedChunk


def _hit(src: str, idx: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=idx,
        page_content=text,
        metadata={"source_name": src, "chunk_id": f"{src}_{idx}"},
        distance=0.5,
    )


def test_diversify_interleaves_sources():
    hits = [
        _hit("a.txt", 0, "a0"),
        _hit("a.txt", 1, "a1"),
        _hit("b.txt", 2, "b0"),
        _hit("b.txt", 3, "b1"),
    ]
    out = diversify_chunks_round_robin(hits, limit=4)
    assert out[0].metadata.get("source_name") != out[1].metadata.get("source_name")
    names = [h.metadata.get("source_name") for h in out]
    assert "a.txt" in names and "b.txt" in names


def test_select_generation_context_diversifies_when_broad_and_multi_file():
    ranked = [
        _hit("playbook_long.txt", 0, "ZEPHYR"),
        _hit("playbook_long.txt", 1, "more playbook"),
        _hit("corp_finance.txt", 2, "Acme revenue"),
    ]
    ctx = select_generation_context(
        ranked,
        mode="qa",
        top_k=4,
        nvec=10,
        broad_document_question=True,
    )
    assert len(ctx) >= 2
    srcs = [str(h.metadata.get("source_name")) for h in ctx]
    assert "corp_finance.txt" in srcs


def test_select_generation_context_skips_diversify_for_section_navigation():
    ranked = [
        _hit("playbook_long.txt", 0, "anchor section seven disaster"),
        _hit("corp_finance.txt", 1, "Acme revenue"),
        _hit("playbook_long.txt", 2, "appendix filler"),
    ]
    ctx = select_generation_context(
        ranked,
        mode="qa",
        top_k=4,
        nvec=10,
        broad_document_question=True,
        section_navigation_query=True,
    )
    assert ctx[0].page_content == "anchor section seven disaster"
