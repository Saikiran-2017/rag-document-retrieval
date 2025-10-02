"""
Grounded answer generation: answer user questions using only retrieved context.

Uses a strict system prompt and low temperature to reduce hallucination. Citations
reference [SOURCE n] labels that match the numbered context blocks.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_openai_api_key
from app.llm.answer_validation import validate_grounded_answer, validate_web_markdown_links
from app.services.message_service import merge_notes
from app.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

_WEB_BLOCK_URL_RE = re.compile(r"^URL:\s*(\S+)", re.MULTILINE)


def _urls_from_web_context_block(web_context_block: str) -> list[str]:
    return _WEB_BLOCK_URL_RE.findall(web_context_block or "")

DEFAULT_CHAT_MODEL = "gpt-4o-mini"

# FAISS L2 distance on retrieved chunks: lower is closer. Above this, treat as weak match.
USEFUL_RETRIEVAL_MAX_L2 = 1.18
# Hybrid (BM25 + vector + RRF): stricter gates — prefer web/general over weak doc grounding.
_HYBRID_MAX_L2 = 1.02
_HYBRID_MIN_RRF = 0.014
_HYBRID_VERY_GOOD_L2 = 0.82
# Document tasks (summarize / compare) allow slightly looser top-hit retrieval.
_HYBRID_TASK_MAX_L2 = 1.1
_HYBRID_TASK_MIN_RRF = 0.012
_HYBRID_TASK_VERY_GOOD_L2 = 0.9

UNKNOWN_PHRASE = "I don't know based on the provided documents."

GENERAL_ASSISTANT_PROMPT = """You are the assistant in a chat app. Reply in a clear, friendly tone.

How to write:
- Answer the user's message directly. Prefer short paragraphs; use a short bullet list only when it improves clarity (steps, options, or multiple items).
- Be concise by default; add detail only when the question needs it.
- Use plain language. Light Markdown is fine (e.g. **bold** for a key term); avoid heavy formatting.

Do not:
- Add citations, source numbers, [SOURCE n], or phrases like "according to your documents" or "based on the file you uploaded". This turn has no document context.
- Invent that you read specific files or quoted material.

If you are unsure or the question is outside your knowledge, say so briefly and suggest what would help (e.g. rephrase, add constraints)."""

GROUNDING_SYSTEM_PROMPT = f"""You are a precise assistant for document Q&A.

Rules:
- Evidence is ONLY the text under the line "Text:" inside each [SOURCE N] block. Ignore any training knowledge.
- If the evidence is insufficient to answer, reply with exactly this sentence and nothing else: {UNKNOWN_PHRASE}
- Do not guess, speculate, or infer facts that are not clearly supported by the Text lines.
- Do not use "probably", "likely", or "typically" unless that exact uncertainty appears in the Text.
- Do not invent [SOURCE N] labels; only use numbers N that appear in the CONTEXT (1 through the highest [SOURCE N] shown).
- When you state a fact from the Text, cite it with the matching label, e.g. [SOURCE 1].
- If two Text passages contradict each other, say so briefly and cite both sources.
- If the question asks for content not present in the Text lines, say so and use the unknown phrase above.
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
    # Optional note from post-generation checks (citation / overlap); surfaced via assistant_note.
    validation_warning: str | None = None


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


# Cap per-chunk text in the prompt for lower latency and token cost (UI unchanged).
_MAX_CONTEXT_CHARS_PER_CHUNK = 2000


def _truncate_chunk_text(text: str, max_chars: int = _MAX_CONTEXT_CHARS_PER_CHUNK) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "..."


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
        body = _truncate_chunk_text(h.page_content)
        blocks.append(
            f"[SOURCE {i}] {_meta_str(m, 'source_name')} · p.{_normalize_page_label(m)}\n"
            f"Text:\n{body}"
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


def _yield_llm_stream(model: ChatOpenAI, messages: list[SystemMessage | HumanMessage]) -> Iterator[str]:
    for chunk in model.stream(messages):
        c = getattr(chunk, "content", None)
        if c:
            if isinstance(c, str):
                yield c
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        yield str(b.get("text", ""))
                    elif isinstance(b, str):
                        yield b


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
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Chat model for RAG answers (temperature 0 by default for faithfulness)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": get_openai_api_key(),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


# Keeps general-chat replies tight for the UI (lower latency vs very long completions).
_GENERAL_ANSWER_MAX_TOKENS = 600

# Caps grounded completion length; answers stay citation-focused.
_GROUNDED_ANSWER_MAX_TOKENS = 900
_DOCUMENT_TASK_MAX_TOKENS = 1200

# Document-centric tasks (summarize / extract / compare): same evidence rules as Q&A, different output shape.
SUMMARIZE_SYSTEM_PROMPT = f"""You summarize text supplied only in [SOURCE N] blocks below. Evidence is ONLY the lines under "Text:" in each block.

Output format (use Markdown):
1. **Overview** — 2–4 sentences.
2. **Key points** — bullet list (3–8 items when the material supports it).
3. If the excerpts clearly do not cover the topic, say so in one short sentence (you may phrase like: {UNKNOWN_PHRASE}).

Rules:
- Cite [SOURCE N] for specific claims taken from the Text. Only use source numbers that appear in CONTEXT.
- Do not invent content not supported by the Text lines. Do not infer beyond what the excerpts state."""

EXTRACT_SYSTEM_PROMPT = f"""You extract structured information from text in [SOURCE N] blocks. Evidence is ONLY the lines under "Text:" in each block.

The user may ask for deadlines, action items, key entities, requirements, decisions, or similar.

Output format (use Markdown):
- Use short headings (###) and bullet lists.
- If a category has no matches, state "None found in the supplied excerpts."
- Cite [SOURCE N] for each non-trivial item where possible.
- If the excerpts are insufficient, say so briefly (you may phrase like: {UNKNOWN_PHRASE})."""

COMPARE_SYSTEM_PROMPT = f"""You compare content across documents using only text in [SOURCE N] blocks. Evidence is ONLY the lines under "Text:" in each block.

Cover what the user asked (e.g. differences, similarities, policy changes, tone) using only the supplied excerpts.
- Prefer a short **Summary** then **Details** with bullets or a small Markdown table when it helps readability.
- Cite [SOURCE N] for claims tied to specific passages.
- If excerpts come from only one document or are insufficient to compare, say so clearly (you may phrase like: {UNKNOWN_PHRASE})."""


def build_document_task_messages(
    task: str,
    query: str,
    context_block: str,
) -> list[SystemMessage | HumanMessage]:
    """System + user messages for summarize / extract / compare (single completion, no agent loop)."""
    prompts = {
        "summarize": SUMMARIZE_SYSTEM_PROMPT,
        "extract": EXTRACT_SYSTEM_PROMPT,
        "compare": COMPARE_SYSTEM_PROMPT,
    }
    system = prompts.get(task)
    if not system:
        raise ValueError(f"Unknown document task: {task!r}")
    user_body = (
        f"CONTEXT:\n{context_block}\n\n"
        f"USER REQUEST:\n{query.strip()}\n\n"
        "Follow the system instructions. Use only the Text under each [SOURCE N]."
    )
    return [SystemMessage(content=system), HumanMessage(content=user_body)]


def generate_document_task_answer(
    task: str,
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    llm: ChatOpenAI | None = None,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
) -> GroundedAnswer:
    """
    One-shot document task (summarize / extract / compare) over retrieved chunks.

    Same citation numbering as grounded Q&A; empty chunks yields a safe fixed message without calling the model.
    """
    if task not in ("summarize", "extract", "compare"):
        raise ValueError(f"Unknown document task: {task!r}")
    if not query.strip():
        raise ValueError("Query must be non-empty.")

    sources = chunks_to_source_refs(retrieved_chunks)
    if not retrieved_chunks:
        return GroundedAnswer(
            answer="No document excerpts were available for this task. Add and sync documents, then try again.",
            sources=(),
        )

    context_block = format_context_for_prompt(retrieved_chunks)
    messages = build_document_task_messages(task, query, context_block)
    model = llm or create_chat_llm(
        model=chat_model,
        temperature=temperature,
        max_tokens=_DOCUMENT_TASK_MAX_TOKENS,
    )
    logger.info("Calling chat model for document task %r (%s chunk(s))", task, len(retrieved_chunks))
    response = model.invoke(messages)
    answer = _coerce_text_content(response.content).strip()
    if not answer:
        answer = UNKNOWN_PHRASE
    fixed, warn = validate_grounded_answer(answer, retrieved_chunks, unknown_phrase=UNKNOWN_PHRASE)
    return GroundedAnswer(answer=fixed, sources=sources, validation_warning=warn)


def generate_general_answer(
    query: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.35,
) -> str:
    """
    Plain assistant reply without retrieved context or citations.
    Same OpenAI chat stack as grounded answers; tuned for concise, natural chat.
    """
    if not query.strip():
        raise ValueError("Query must be non-empty.")

    model = create_chat_llm(
        model=chat_model,
        temperature=temperature,
        max_tokens=_GENERAL_ANSWER_MAX_TOKENS,
    )
    messages = [
        SystemMessage(content=GENERAL_ASSISTANT_PROMPT),
        HumanMessage(content=query.strip()),
    ]
    logger.info("Calling chat model for general answer (no retrieval)")
    response = model.invoke(messages)
    out = _coerce_text_content(response.content).strip()
    return out if out else "I don't have a response right now."


def retrieval_is_useful(
    hits: list[RetrievedChunk],
    *,
    max_l2_distance: float = USEFUL_RETRIEVAL_MAX_L2,
) -> bool:
    """
    Simple gate for whether to use document grounding: need at least one hit whose
    FAISS L2 distance is at or below the threshold (best hit is hits[0]).
    """
    if not hits:
        return False
    return float(hits[0].distance) <= float(max_l2_distance)


def hybrid_retrieval_is_useful(
    hits: list[RetrievedChunk],
    *,
    for_document_task: bool = False,
) -> bool:
    """
    Gate after hybrid RRF + rerank: require vector similarity and fusion mass,
    unless the vector match is very strong (distance alone).
    """
    if not hits:
        return False
    h = hits[0]
    d = float(h.distance)
    rrf = float(h.metadata.get("rrf_score", 0.0) or 0.0)
    if for_document_task:
        max_l2 = _HYBRID_TASK_MAX_L2
        min_rrf = _HYBRID_TASK_MIN_RRF
        very_good = _HYBRID_TASK_VERY_GOOD_L2
    else:
        max_l2 = _HYBRID_MAX_L2
        min_rrf = _HYBRID_MIN_RRF
        very_good = _HYBRID_VERY_GOOD_L2
    if rrf > 0:
        strong_vec = d <= very_good
        fusion_ok = d <= max_l2 and rrf >= min_rrf
        return fusion_ok or strong_vec
    return d <= USEFUL_RETRIEVAL_MAX_L2


# Stricter than hybrid_retrieval_is_useful: when a file is only partially trusted
# (ready_limited) or the library has no fully healthy file, require a clearly strong match.
_LIMITED_QA_MAX_L2 = 0.74
_LIMITED_QA_MIN_RRF = 0.016
_LIMITED_TASK_MAX_L2 = 0.92
_LIMITED_TASK_MIN_RRF = 0.013


def hybrid_hit_strong_for_limited_corpora(
    hit: RetrievedChunk,
    *,
    for_document_task: bool = False,
) -> bool:
    """True when the top hit is strong enough to ground answers for weak-trust documents."""
    d = float(hit.distance)
    rrf = float(hit.metadata.get("rrf_score", 0.0) or 0.0)
    if for_document_task:
        return d <= _LIMITED_TASK_MAX_L2 and rrf >= _LIMITED_TASK_MIN_RRF
    if rrf > 0:
        return d <= _LIMITED_QA_MAX_L2 and rrf >= _LIMITED_QA_MIN_RRF
    return d <= USEFUL_RETRIEVAL_MAX_L2


WEB_GROUNDED_SYSTEM_PROMPT = """You are a careful assistant. You answer ONLY using the numbered WEB snippets below.

Each block is [WEB n] with Title, URL, and Snippet text.

Rules:
- Use ONLY information supported by the Snippet lines. Do not invent facts.
- Every factual claim that comes from a snippet must include a Markdown link. The URL in parentheses must match a URL from that same [WEB n] block exactly (copy-paste; no edits).
- Do NOT invent URLs, domains, or sources that do not appear in the WEB blocks.
- If snippets are insufficient, say so briefly and list what is missing.
- Keep the answer concise and avoid repeating the same sentence."""


BLENDED_SYSTEM_PROMPT = f"""You answer using two evidence types that may both appear:
1) Document passages under [SOURCE n] — Text: lines only.
2) Web snippets under [WEB n] — Snippet: lines only.

Rules:
- Prefer SOURCE passages for anything covered in your documents; do not duplicate the same fact from WEB when SOURCE already states it.
- Use WEB snippets only for time-sensitive or clearly external facts not in SOURCE, each as [title](url) with the EXACT URL from that WEB block.
- Do not invent [SOURCE n] or [WEB n] labels; only use numbers that appear.
- Cite [SOURCE n] for document claims; keep web claims clearly separate (you may label with "Web:" for one sentence if helpful).
- If document evidence is insufficient for part of the question, say so and rely on WEB only for that part if available.
- If you cannot answer from either, say so briefly (you may use wording like: {UNKNOWN_PHRASE} for the document-only portion)."""


def build_web_grounded_messages(query: str, web_block: str) -> list[SystemMessage | HumanMessage]:
    user_body = (
        f"WEB SNIPPETS:\n{web_block}\n\nQUESTION:\n{query.strip()}\n\n"
        "Answer using only the snippets. Link every web-sourced claim."
    )
    return [SystemMessage(content=WEB_GROUNDED_SYSTEM_PROMPT), HumanMessage(content=user_body)]


def build_blended_messages(query: str, doc_block: str, web_block: str) -> list[SystemMessage | HumanMessage]:
    user_body = (
        f"DOCUMENT CONTEXT:\n{doc_block}\n\n---\n\nWEB SNIPPETS:\n{web_block}\n\n"
        f"QUESTION:\n{query.strip()}\n\n"
        "Follow the system instructions for citing SOURCE and WEB evidence."
    )
    return [SystemMessage(content=BLENDED_SYSTEM_PROMPT), HumanMessage(content=user_body)]


def generate_web_grounded_answer(
    query: str,
    web_context_block: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> str:
    if not query.strip():
        raise ValueError("Query must be non-empty.")
    if not web_context_block.strip() or "No web results" in web_context_block:
        return "No web results were available for this query. Try rephrasing or check your connection."
    model = create_chat_llm(model=chat_model, temperature=0.2, max_tokens=_GROUNDED_ANSWER_MAX_TOKENS)
    messages = build_web_grounded_messages(query, web_context_block)
    response = model.invoke(messages)
    return _coerce_text_content(response.content).strip() or "No answer generated."


def generate_blended_answer(
    query: str,
    doc_chunks: list[RetrievedChunk],
    web_context_block: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> GroundedAnswer:
    if not query.strip():
        raise ValueError("Query must be non-empty.")
    doc_block = format_context_for_prompt(doc_chunks)
    model = create_chat_llm(model=chat_model, temperature=0.0, max_tokens=_GROUNDED_ANSWER_MAX_TOKENS)
    messages = build_blended_messages(query, doc_block, web_context_block)
    response = model.invoke(messages)
    answer = _coerce_text_content(response.content).strip() or UNKNOWN_PHRASE
    sources = chunks_to_source_refs(doc_chunks)
    fixed, warn = validate_grounded_answer(answer, doc_chunks, unknown_phrase=UNKNOWN_PHRASE)
    wfixed, wwarn = validate_web_markdown_links(fixed, _urls_from_web_context_block(web_context_block))
    return GroundedAnswer(
        answer=wfixed,
        sources=sources,
        validation_warning=merge_notes(warn, wwarn),
    )


def stream_grounded_answer_tokens(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> Iterator[str]:
    """Yield text tokens for Streamlit ``st.write_stream`` (document-grounded only)."""
    if not retrieved_chunks:
        yield "No passages retrieved."
        return
    context_block = format_context_for_prompt(retrieved_chunks)
    messages = build_grounded_messages(query, context_block)
    model = create_chat_llm(
        model=chat_model,
        temperature=0.0,
        max_tokens=_GROUNDED_ANSWER_MAX_TOKENS,
    )
    yield from _yield_llm_stream(model, messages)


def stream_general_answer_tokens(
    query: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.35,
) -> Iterator[str]:
    if not query.strip():
        yield ""
        return
    model = create_chat_llm(
        model=chat_model,
        temperature=temperature,
        max_tokens=_GENERAL_ANSWER_MAX_TOKENS,
    )
    messages = [
        SystemMessage(content=GENERAL_ASSISTANT_PROMPT),
        HumanMessage(content=query.strip()),
    ]
    yield from _yield_llm_stream(model, messages)


def stream_web_grounded_answer_tokens(
    query: str,
    web_context_block: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> Iterator[str]:
    if not query.strip():
        yield ""
        return
    if not web_context_block.strip() or "No web results" in web_context_block:
        yield "No web results were available for this query. Try rephrasing or check your connection."
        return
    model = create_chat_llm(model=chat_model, temperature=0.2, max_tokens=_GROUNDED_ANSWER_MAX_TOKENS)
    messages = build_web_grounded_messages(query, web_context_block)
    yield from _yield_llm_stream(model, messages)


def stream_blended_answer_tokens(
    query: str,
    doc_chunks: list[RetrievedChunk],
    web_context_block: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> Iterator[str]:
    if not query.strip():
        yield ""
        return
    doc_block = format_context_for_prompt(doc_chunks)
    model = create_chat_llm(model=chat_model, temperature=0.0, max_tokens=_GROUNDED_ANSWER_MAX_TOKENS)
    messages = build_blended_messages(query, doc_block, web_context_block)
    yield from _yield_llm_stream(model, messages)


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
    Grounding reduces hallucination but is not mathematically guaranteed; always review for production.
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
    model = llm or create_chat_llm(
        model=chat_model,
        temperature=temperature,
        max_tokens=_GROUNDED_ANSWER_MAX_TOKENS,
    )

    logger.info("Calling chat model for grounded answer (%s chunk(s) in context)", len(retrieved_chunks))
    response = model.invoke(messages)
    answer = _coerce_text_content(response.content).strip()

    if not answer:
        answer = UNKNOWN_PHRASE
    fixed, warn = validate_grounded_answer(answer, retrieved_chunks, unknown_phrase=UNKNOWN_PHRASE)
    return GroundedAnswer(answer=fixed, sources=sources, validation_warning=warn)


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

# Optimized 2025-09-06

# Optimized 2025-10-01

# Optimized 2025-10-21

# Optimized 2025-08-12

# Optimized 2025-09-06
