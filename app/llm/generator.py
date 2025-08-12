"""
Grounded answer generation: answer user questions using only retrieved context.

Uses a strict system prompt and low temperature to reduce hallucination. Citations
reference [SOURCE n] labels that match the numbered context blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_openai_api_key
from app.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "gpt-4o-mini"

UNKNOWN_PHRASE = "I don't know based on the provided documents."

GROUNDING_SYSTEM_PROMPT = f"""You are a precise assistant for document Q&A.

Rules:
- Evidence is ONLY the text under the line "Text:" inside each [SOURCE N] block. Ignore any training knowledge.
- If the evidence is insufficient to answer, reply with exactly this sentence and nothing else: {UNKNOWN_PHRASE}
- Do not guess, speculate, or infer facts that are not clearly supported by the Text lines.
- Do not invent [SOURCE N] labels; only use numbers N that appear in the CONTEXT.
- When you state a fact from the Text, cite it with the matching label, e.g. [SOURCE 1].
- If two Text passages contradict each other, say so briefly and cite both sources.
- Keep the answer concise and directly responsive to the question."""


@dataclass(frozen=True)
class SourceRef:
    """
    One retrieved passage included in the prompt.

    ``source_number`` matches [SOURCE N] in the prompt (1-based). Safe to render in Streamlit
    as citations alongside ``result.answer``.
    """

    source_number: int
    chunk_id: str
    source_name: str
    page_label: str
    file_path: str


@dataclass(frozen=True)
class GroundedAnswer:
    """Model output plus the sources that were supplied as context (UI-ready, immutable)."""

    answer: str
    sources: tuple[SourceRef, ...]


def _normalize_page_label(meta: dict[str, Any]) -> str:
    p = meta.get("page_number")
    if p is None or p == "":
        return "N/A"
    return str(p)


def _meta_str(meta: dict[str, Any], key: str) -> str:
    v = meta.get(key)
    return "" if v is None else str(v)


def chunks_to_source_refs(chunks: list[RetrievedChunk]) -> tuple[SourceRef, ...]:
    """Build citation rows from retrieved chunks (``source_number`` aligns with [SOURCE N])."""
    refs: list[SourceRef] = []
    for i, h in enumerate(chunks, start=1):
        m = h.metadata
        refs.append(
            SourceRef(
                source_number=i,
                chunk_id=_meta_str(m, "chunk_id"),
                source_name=_meta_str(m, "source_name"),
                page_label=_normalize_page_label(m),
                file_path=_meta_str(m, "file_path"),
            )
        )
    return tuple(refs)


def format_context_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Turn retrieved chunks into numbered context blocks the model must stay within.

    Only lines under ``Text:`` are treated as quotable evidence in the system instructions.
    """
    if not chunks:
        return "(No context passages were retrieved.)"

    blocks: list[str] = []
    for i, h in enumerate(chunks, start=1):
        m = h.metadata
        blocks.append(
            f"[SOURCE {i}]\n"
            f"chunk_id: {_meta_str(m, 'chunk_id')}\n"
            f"source_name: {_meta_str(m, 'source_name')}\n"
            f"file_path: {_meta_str(m, 'file_path')}\n"
            f"page_number: {_normalize_page_label(m)}\n"
            f"Text:\n{h.page_content.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def build_grounded_messages(query: str, context_block: str) -> list[SystemMessage | HumanMessage]:
    """System + user messages for a single grounded completion."""
    user_body = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query.strip()}\n\n"
        "Answer using only the Text under each [SOURCE N]. "
        "Cite [SOURCE N] when you use a passage. "
        f"If you cannot answer from the Text, reply exactly: {UNKNOWN_PHRASE}"
    )
    return [SystemMessage(content=GROUNDING_SYSTEM_PROMPT), HumanMessage(content=user_body)]


def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content)


def create_chat_llm(
    *,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Chat model for RAG answers (temperature 0 by default for faithfulness)."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=get_openai_api_key(),
    )


def generate_grounded_answer(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    llm: ChatOpenAI | None = None,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
) -> GroundedAnswer:
    """
    Produce an answer that uses only ``retrieved_chunks`` as evidence.

    If ``retrieved_chunks`` is empty, skips the LLM and returns a safe fixed message.

    ``sources`` lists every chunk in the prompt; ``source_number`` matches [SOURCE N].
    Grounding reduces hallucination but is not mathematically guaranteed—always review for production.
    """
    if not query.strip():
        raise ValueError("Query must be non-empty.")

    sources = chunks_to_source_refs(retrieved_chunks)

    if not retrieved_chunks:
        return GroundedAnswer(
            answer="No relevant passages were retrieved from the knowledge base. " + UNKNOWN_PHRASE,
            sources=(),
        )

    context_block = format_context_for_prompt(retrieved_chunks)
    messages = build_grounded_messages(query, context_block)
    model = llm or create_chat_llm(model=chat_model, temperature=temperature)

    logger.info("Calling chat model for grounded answer (%s chunk(s) in context)", len(retrieved_chunks))
    response = model.invoke(messages)
    answer = _coerce_text_content(response.content).strip()

    if not answer:
        answer = UNKNOWN_PHRASE

    return GroundedAnswer(answer=answer, sources=sources)


def print_grounded_result(result: GroundedAnswer, *, query: str) -> None:
    """Print answer and source table for terminal demos."""
    print(f"Question: {query!r}\n")
    print("Answer:")
    print(result.answer)
    print("\nSources in context (supplied to the model):")
    if not result.sources:
        print("  (none)")
    else:
        for s in result.sources:
            fp = s.file_path or "(n/a)"
            print(
                f"  - [SOURCE {s.source_number}] chunk_id={s.chunk_id!r} "
                f"file={s.source_name!r} page={s.page_label!r} path={fp!r}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from app.ingestion.loader import load_raw_directory
    from app.retrieval.vector_store import (
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_INDEX_NAME,
        build_faiss_from_chunks,
        create_openai_embeddings,
        faiss_index_files_exist,
        faiss_vector_count,
        get_default_faiss_folder,
        load_faiss_index,
        retrieve_top_k,
        save_faiss_index,
    )
    from app.utils.chunker import chunk_ingested_documents

    folder = get_default_faiss_folder()
    index_name = DEFAULT_INDEX_NAME
    embed_model = DEFAULT_EMBEDDING_MODEL

    print("=== Phase 6 demo: ingest -> chunk -> FAISS -> retrieve -> grounded answer ===\n")

    raw_docs = load_raw_directory()
    chunks = chunk_ingested_documents(raw_docs, chunk_size=500, chunk_overlap=80)
    if not chunks:
        print("No chunks produced. Add files under data/raw/ and run again.")
        raise SystemExit(1)

    embeddings = create_openai_embeddings(model=embed_model)
    if faiss_index_files_exist(folder, index_name=index_name):
        store = load_faiss_index(
            folder_path=folder,
            index_name=index_name,
            embeddings=embeddings,
        )
        print("Loaded existing FAISS index.\n")
    else:
        store = build_faiss_from_chunks(chunks, embeddings=embeddings)
        save_faiss_index(store, folder_path=folder, index_name=index_name)
        print("Built and saved FAISS index.\n")

    question = "What does the sample text say about chunking and metadata?"
    k = min(4, faiss_vector_count(store))
    hits = retrieve_top_k(store, question, k=k)
    print(f"Retrieved {len(hits)} chunk(s).\n")

    result = generate_grounded_answer(question, hits)
    print_grounded_result(result, query=question)

# Optimized 2024-08-12

# Optimized 2024-09-06

# Optimized 2024-10-01

# Optimized 2024-10-21

# Optimized 2024-08-12

# Optimized 2024-09-06

# Optimized 2025-08-12
