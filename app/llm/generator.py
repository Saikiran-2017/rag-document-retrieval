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


def _remove_emdashes(text: str) -> str:
    """Remove em-dashes from text, replacing with commas for readability."""
    if not text:
        return text
    # Replace em-dashes (—) with commas for clean output
    return text.replace("—", ",")

DEFAULT_CHAT_MODEL = "gpt-4o-mini"

# FAISS L2 distance on retrieved chunks: lower is closer. Above this, treat as weak match.
USEFUL_RETRIEVAL_MAX_L2 = 1.22
# Hybrid (BM25 + vector + RRF): balance precision vs recall for doc QA (broad questions sit higher in L2).
_HYBRID_MAX_L2 = 1.16
_HYBRID_MIN_RRF = 0.0085
_HYBRID_VERY_GOOD_L2 = 0.88
# Document tasks (summarize / compare) allow slightly looser top-hit retrieval.
_HYBRID_TASK_MAX_L2 = 1.1
_HYBRID_TASK_MIN_RRF = 0.012
_HYBRID_TASK_VERY_GOOD_L2 = 0.9
# Broad overview / vague doc-shaped QA: top hit L2 is often higher; RRF still shows keyword fit.
_HYBRID_BROAD_QA_MAX_L2 = 1.4
_HYBRID_BROAD_QA_MIN_RRF = 0.006
_HYBRID_BROAD_QA_VERY_GOOD_L2 = 0.96
_HYBRID_BROAD_QA_HIGH_RRF = 0.02
_HYBRID_BROAD_QA_MAX_L2_WHEN_HIGH_RRF = 1.55
# Sparse entity / role lookups (CFO name, etc.): top L2 often slightly above strict QA but RRF is strong.
_HYBRID_LOOKUP_MAX_L2 = 1.32
_HYBRID_LOOKUP_MIN_RRF = 0.008
_HYBRID_LOOKUP_VERY_GOOD_L2 = 0.88
_HYBRID_LOOKUP_HIGH_RRF = 0.026
_HYBRID_LOOKUP_MAX_L2_WHEN_HIGH_RRF = 1.45

UNKNOWN_PHRASE = "I don't know based on the provided documents."

GENERAL_ASSISTANT_PROMPT = """You are the assistant in a document Q&A and knowledge-workspace chat app. Reply in a clear, friendly tone.

Domain vocabulary (when the user asks about these terms in this app, prefer the information-retrieval / ML meaning):
- **RAG** means **Retrieval-Augmented Generation**: combining search/retrieval over a knowledge source with a language model so answers can be grounded in that material.
- **Retrieval-augmented generation** is the same idea as RAG (full phrase).
- **Embeddings** are numeric vector representations of text used for semantic search and similarity.
- **Vector database** (vector store) indexes those vectors to find nearest neighbors for retrieval.

How to write:
- Answer the user's message directly. Prefer short paragraphs; use a short bullet list only when it improves clarity (steps, options, or multiple items).
- Be concise by default; add detail only when the question needs it.
- Use plain language. Light Markdown is fine (e.g. **bold** for a key term); avoid heavy formatting.

Do not:
- Add citations, source numbers, [SOURCE n], or phrases like "according to your documents" or "based on the file you uploaded". This turn has no document context.
- Invent that you read specific files or quoted material.

Assistant / product identity (this turn has no document context):
- If the user asks who you are, your name, who built you, or what this assistant is: answer as **Knowledge Assistant**, the chat helper for this workspace. Do not role-play as a person from hypothetical uploads and do not invent vendor or author names.
- Briefly clarify you answer from the user's library when retrieval applies, and otherwise from general knowledge.

If you are unsure or the question is outside your knowledge, say so briefly and suggest what would help (e.g. rephrase, add constraints)."""

# Used when the user clearly asked about their library but retrieval did not surface
# reliable passages (or web is disabled/thin). Prevents unrelated general knowledge answers.
DOCUMENT_ABSTAIN_GENERAL_PROMPT = f"""You are a careful assistant in a document Q&A app.

The user is asking about their uploaded library, but no reliable passages were retrieved for this question (or search could not confirm a match).

Rules:
- Reply in one or two short sentences.
- Say clearly that you cannot find this in their documents / uploaded materials. You may use wording like: {UNKNOWN_PHRASE}
- Do NOT answer using outside knowledge: no recipes, tutorials, trivia, or invented facts that pretend to come from their files.
- Do NOT fabricate citations, file names, or quotes.
- If the question is impossible to answer without external knowledge, say their materials do not contain it and stop.

Tone: neutral and helpful; no guilt-tripping."""

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
- If the question names a specific person, company, product, or program (for example a CEO, employer, rocket program, or product name) and the Text does not state the requested fact, use the unknown phrase. Do not fill in with outside news, statistics, or general knowledge about that entity.
- For overview, summary, main themes, or "what is this document about" questions, synthesize substantive meaning from the Text (people, organizations, roles, goals, projects, products, timelines, outcomes) across all SOURCE blocks you were given. Do not answer with a catalogue of form field names, metadata labels, or generic claims that the file is only a "structured record" or "labeled form" unless the Text itself says so. Prefer an opening like "This document provides an overview of …" and cite multiple [SOURCE n] when different passages support different parts.
- When CONTEXT includes multiple [SOURCE N] blocks and more than one is relevant, use more than one citation; do not answer from only the first block if other blocks clearly apply.
- If the question targets a named section, heading topic, or a specific role/title (e.g. CFO), a single explicit sentence in the Text that states the fact is sufficient: answer briefly and cite that source. Do not abstain only because the passage is short, when it clearly names the section/topic or the person/role asked about.
- If the question is extremely vague (e.g. "what is this", "explain this") and the Text does not clearly identify a single subject, either ask one short clarifying question or reply with the unknown phrase above—do not invent specifics.
- If the user asks for salary, SSN/Social Security number, passwords, secrets, or similarly sensitive personal details that do not appear in the Text, reply with the unknown phrase above—do not guess from general knowledge.
- Keep the answer concise and directly responsive to the question.

Voice (who is "you"):
- You are the app's assistant, not a human described in the documents. Never answer in first person as if you were a named person, employer, or applicant from the Text (for example, do not write "I am …" for a person in a resume or form unless the Text itself is written in second person to the reader and you are clearly reporting that wording).
- Attribute facts about people or organizations to the document: prefer phrasing like "The document states …", "According to [SOURCE n] …", or "Based on the uploaded file …" before giving the fact. Do not present document facts as your own autobiography."""


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
    # Optional UI helpers (safe for older clients to ignore).
    source_label: str = ""
    snippet: str = ""


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


def _source_label(source_name: str, page_label: str) -> str:
    sn = (source_name or "").strip()
    if sn.lower().endswith((".pdf", ".txt", ".docx")):
        sn = sn.rsplit(".", 1)[0]
    pl = (page_label or "").strip()
    if not pl or pl.lower() in ("n/a", "none", "-"):
        return sn or "Source"
    return f"{sn or 'Source'} · p.{pl}"


def _snippet(text: str, *, limit: int = 220) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = " ".join(t.split())
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


def chunks_to_source_refs(chunks: list[RetrievedChunk]) -> tuple[SourceRef, ...]:
    """Build citation rows from retrieved chunks (``source_number`` aligns with [SOURCE N])."""
    refs: list[SourceRef] = []
    for i, h in enumerate(chunks, start=1):
        m = h.metadata
        src = _meta_str(m, "source_name")
        pl = _normalize_page_label(m)
        refs.append(
            SourceRef(
                source_number=i,
                chunk_id=_meta_str(m, "chunk_id"),
                source_name=src,
                page_label=pl,
                file_path=_meta_str(m, "file_path"),
                source_label=_source_label(src, pl),
                snippet=_snippet(h.page_content),
            )
        )
    return tuple(refs)


# Cap per-chunk text in the prompt for lower latency and token cost (UI unchanged).
_MAX_CONTEXT_CHARS_PER_CHUNK = 2000


_CTX_STOP = frozenset(
    "a an the and or but if to of in on for with as at by from is are was were be been being "
    "it this that these those i you we they he she not no yes so than then there here do does did "
    "can could should would will just only very more most also into about out up what which who "
    "how when where why all any each both few such than too very my your our their".split()
)


def _query_keywords(query: str, *, limit: int = 8) -> list[str]:
    """Cheap keyword extraction for context slicing (no extra model calls)."""
    q = (query or "").lower()
    toks = [t for t in re.findall(r"[a-z0-9]{3,}", q) if t not in _CTX_STOP]
    # Stable de-dup
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _slice_text_around_match(text: str, query: str, *, max_chars: int) -> str:
    """
    Keep evidence near the query match even when the chunk is long.

    Strategy:
    - If we can find a keyword match, return a window around the first match.
    - Otherwise, include head + tail so late facts aren't silently dropped.
    """
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    kw = _query_keywords(query)
    low = t.lower()
    hit_at: int | None = None
    for k in kw:
        i = low.find(k)
        if i >= 0:
            hit_at = i
            break
    if hit_at is not None:
        half = max_chars // 2
        start = max(0, hit_at - half)
        end = min(len(t), start + max_chars)
        start = max(0, end - max_chars)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(t) else ""
        return prefix + t[start:end].strip() + suffix
    # Fallback: head + tail.
    head = max(200, int(max_chars * 0.6))
    tail = max(120, max_chars - head - 20)
    head_txt = t[:head].rstrip()
    tail_txt = t[-tail:].lstrip()
    return f"{head_txt}\n...\n{tail_txt}"


def _truncate_chunk_text(text: str, *, query: str | None = None, max_chars: int = _MAX_CONTEXT_CHARS_PER_CHUNK) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    if query and query.strip():
        return _slice_text_around_match(t, query, max_chars=max_chars)
    return t[: max_chars - 1].rstrip() + "..."


def format_context_for_prompt(chunks: list[RetrievedChunk], *, query: str | None = None) -> str:
    """
    Turn retrieved chunks into numbered context blocks the model must stay within.

    Only lines under ``Text:`` are treated as quotable evidence in the system instructions.
    """
    if not chunks:
        return "(No context passages were retrieved.)"

    blocks: list[str] = []
    for i, h in enumerate(chunks, start=1):
        m = h.metadata
        body = _truncate_chunk_text(h.page_content, query=query)
        blocks.append(
            f"[SOURCE {i}] {_meta_str(m, 'source_name')} · p.{_normalize_page_label(m)}\n"
            f"Text:\n{body}"
        )
    return "\n\n---\n\n".join(blocks)


def build_grounded_messages(
    query: str,
    context_block: str,
    *,
    n_context_sources: int = 1,
    section_navigation_query: bool = False,
    broad_document_summary: bool = False,
) -> list[SystemMessage | HumanMessage]:
    """System + user messages for a single grounded completion."""
    synth = ""
    if n_context_sources >= 2:
        synth = (
            "\nYou were given multiple SOURCE blocks. If the question is broad or spans topics, "
            "draw from every relevant block and cite each one you use (not only [SOURCE 1])."
        )
    summary_note = ""
    if broad_document_summary and not section_navigation_query:
        summary_note = (
            "\nThis question asks for a summary or high-level overview. Write one cohesive answer (you may use a short opening "
            'sentence such as "This document provides an overview of …"). Focus on real substance from the Text: '
            "people, companies, roles, mission or purpose, major projects or products, and concrete facts. "
            "Avoid inventorying form labels, empty field names, or describing the document only as a layout of boxes or tables."
        )
    section_note = ""
    if section_navigation_query:
        section_note = (
            "\nThe question targets a specific section, heading, or labeled topic. "
            "If any Text line explicitly names that section number, heading, anchor, or topic "
            "(even a single sentence), answer from that line and cite the SOURCE. "
            f"Use {UNKNOWN_PHRASE!r} only when no Text line mentions the section/topic asked about."
        )
    if section_navigation_query:
        answer_rules = (
            "Answer using only the Text under each [SOURCE N]. Cite [SOURCE N] when you state a fact. "
            f"If (and only if) no Text line is about the section/topic asked, reply exactly: {UNKNOWN_PHRASE}"
        )
    else:
        answer_rules = (
            "Answer using only the Text under each [SOURCE N]. "
            "Cite [SOURCE N] when you use a passage. "
            f"If you cannot answer from the Text, reply exactly: {UNKNOWN_PHRASE}"
        )
    system = GROUNDING_SYSTEM_PROMPT
    if section_navigation_query:
        system = (
            f"{GROUNDING_SYSTEM_PROMPT}\n\n"
            "For section/heading/topic questions: a single explicit sentence that names the section "
            "or topic (e.g. that section seven discusses disaster recovery) is a complete answer; "
            "quote or paraphrase it and cite. Do not reply with the unknown phrase when such a sentence exists."
        )
    elif broad_document_summary:
        system = (
            f"{GROUNDING_SYSTEM_PROMPT}\n\n"
            "For this turn, prioritize a readable narrative summary of what the excerpts convey about the subject matter, "
            "not a description of document structure or field layout."
        )
    user_body = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query.strip()}\n"
        f"{section_note}{summary_note}\n\n"
        f"{answer_rules}{synth}"
    )
    return [SystemMessage(content=system), HumanMessage(content=user_body)]


def _yield_llm_stream(model: ChatOpenAI, messages: list[SystemMessage | HumanMessage]) -> Iterator[str]:
    for chunk in model.stream(messages):
        c = getattr(chunk, "content", None)
        if c:
            if isinstance(c, str):
                yield _remove_emdashes(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        yield _remove_emdashes(str(b.get("text", "")))
                    elif isinstance(b, str):
                        yield _remove_emdashes(b)


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
1. **Overview**: 2–4 sentences that explain what the material is about in plain language (who, what organization, roles, purpose, main initiatives or products, and any notable outcomes). Do not describe the file as a list of blank fields or a "structured form" unless the Text explicitly says that.
2. **Key points**: bullet list (3–8 items when the material supports it); each bullet should capture meaning (facts, themes, decisions), not just label names.
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

    context_block = format_context_for_prompt(retrieved_chunks, query=query)
    messages = build_document_task_messages(task, query, context_block)
    model = llm or create_chat_llm(
        model=chat_model,
        temperature=temperature,
        max_tokens=_DOCUMENT_TASK_MAX_TOKENS,
    )
    logger.info("Calling chat model for document task %r (%s chunk(s))", task, len(retrieved_chunks))
    response = model.invoke(messages)
    answer = _remove_emdashes(_coerce_text_content(response.content).strip())
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
    out = _remove_emdashes(_coerce_text_content(response.content).strip())
    return out if out else "I don't have a response right now."


def generate_document_abstain_answer(
    query: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.2,
) -> str:
    """Short refusal when document grounding was expected but retrieval is weak or empty."""
    if not query.strip():
        raise ValueError("Query must be non-empty.")
    model = create_chat_llm(
        model=chat_model,
        temperature=temperature,
        max_tokens=min(220, _GENERAL_ANSWER_MAX_TOKENS),
    )
    messages = [
        SystemMessage(content=DOCUMENT_ABSTAIN_GENERAL_PROMPT),
        HumanMessage(content=query.strip()),
    ]
    logger.info("Calling chat model for document-scope abstain (weak retrieval)")
    response = model.invoke(messages)
    out = _remove_emdashes(_coerce_text_content(response.content).strip())
    return out if out else UNKNOWN_PHRASE


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
    for_broad_qa: bool = False,
    for_lookup_qa: bool = False,
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
    elif for_broad_qa:
        max_l2 = _HYBRID_BROAD_QA_MAX_L2
        min_rrf = _HYBRID_BROAD_QA_MIN_RRF
        very_good = _HYBRID_BROAD_QA_VERY_GOOD_L2
    elif for_lookup_qa:
        max_l2 = _HYBRID_LOOKUP_MAX_L2
        min_rrf = _HYBRID_LOOKUP_MIN_RRF
        very_good = _HYBRID_LOOKUP_VERY_GOOD_L2
    else:
        max_l2 = _HYBRID_MAX_L2
        min_rrf = _HYBRID_MIN_RRF
        very_good = _HYBRID_VERY_GOOD_L2
    if rrf > 0:
        strong_vec = d <= very_good
        fusion_ok = d <= max_l2 and rrf >= min_rrf
        # Broad/summary questions can have weak vector similarity but very strong hybrid support.
        # If RRF is clearly high, allow a looser distance cap to avoid false refusals on real docs.
        if for_broad_qa and (not fusion_ok) and rrf >= _HYBRID_BROAD_QA_HIGH_RRF:
            fusion_ok = d <= _HYBRID_BROAD_QA_MAX_L2_WHEN_HIGH_RRF
        if for_lookup_qa and (not fusion_ok) and rrf >= _HYBRID_LOOKUP_HIGH_RRF:
            fusion_ok = d <= _HYBRID_LOOKUP_MAX_L2_WHEN_HIGH_RRF
        return fusion_ok or strong_vec
    return d <= USEFUL_RETRIEVAL_MAX_L2


# Stricter than hybrid_retrieval_is_useful: when a file is only partially trusted
# (ready_limited) or the library has no fully healthy file, require a clearly strong match.
_LIMITED_QA_MAX_L2 = 0.92
_LIMITED_QA_MIN_RRF = 0.009
_LIMITED_TASK_MAX_L2 = 0.92
_LIMITED_TASK_MIN_RRF = 0.013
# ready_limited + broad/summary-style QA: allow slightly weaker top vector if fusion mass exists.
_LIMITED_BROAD_QA_MAX_L2 = 1.08
_LIMITED_BROAD_QA_MIN_RRF = 0.006
# When many trusted hits agree on the same source, allow a last-step loosening for ready_limited.
_LIMITED_SAME_SOURCE_COHERE_MIN_HITS = 3
_LIMITED_SAME_SOURCE_COHERE_MAX_L2 = 1.15
_LIMITED_SAME_SOURCE_COHERE_MIN_RRF = 0.005
_LIMITED_QA_COHERE_MIN_HITS = 2
_LIMITED_QA_COHERE_MAX_L2 = 1.02
_LIMITED_LOOKUP_QA_MAX_L2 = 1.1
_LIMITED_LOOKUP_QA_MIN_RRF = 0.006


def hybrid_limited_same_source_fallback(
    hit: RetrievedChunk,
    *,
    same_source_hits_in_window: int,
) -> bool:
    """Extra gate for ready_limited libraries when retrieval clusters on one document."""
    if same_source_hits_in_window < _LIMITED_SAME_SOURCE_COHERE_MIN_HITS:
        return False
    d = float(hit.distance)
    rrf = float(hit.metadata.get("rrf_score", 0.0) or 0.0)
    if rrf <= 0:
        return d <= USEFUL_RETRIEVAL_MAX_L2
    return d <= _LIMITED_SAME_SOURCE_COHERE_MAX_L2 and rrf >= _LIMITED_SAME_SOURCE_COHERE_MIN_RRF


def hybrid_hit_strong_for_limited_corpora(
    hit: RetrievedChunk,
    *,
    for_document_task: bool = False,
    for_broad_qa: bool = False,
    for_lookup_qa: bool = False,
    same_source_hits_in_window: int | None = None,
) -> bool:
    """True when the top hit is strong enough to ground answers for weak-trust documents."""
    d = float(hit.distance)
    rrf = float(hit.metadata.get("rrf_score", 0.0) or 0.0)
    if for_document_task:
        return d <= _LIMITED_TASK_MAX_L2 and rrf >= _LIMITED_TASK_MIN_RRF
    if for_broad_qa:
        if rrf > 0:
            if d <= _LIMITED_BROAD_QA_MAX_L2 and rrf >= _LIMITED_BROAD_QA_MIN_RRF:
                return True
            # Match hybrid_retrieval_is_useful broad band: strong RRF can justify a looser L2 cap.
            if rrf >= _HYBRID_BROAD_QA_HIGH_RRF:
                return d <= _HYBRID_BROAD_QA_MAX_L2_WHEN_HIGH_RRF and rrf >= _LIMITED_BROAD_QA_MIN_RRF
            return False
        return d <= USEFUL_RETRIEVAL_MAX_L2
    if for_lookup_qa:
        if rrf > 0:
            return d <= _LIMITED_LOOKUP_QA_MAX_L2 and rrf >= _LIMITED_LOOKUP_QA_MIN_RRF
        return d <= USEFUL_RETRIEVAL_MAX_L2
    if rrf > 0:
        # Narrow QA sometimes yields only ~2 top hits from one source after chunk/format changes.
        # If the retrieval coheres on the same document, allow a slightly weaker top vector hit.
        if (
            same_source_hits_in_window is not None
            and same_source_hits_in_window >= _LIMITED_QA_COHERE_MIN_HITS
            and d <= _LIMITED_QA_COHERE_MAX_L2
            and rrf >= _LIMITED_QA_MIN_RRF
        ):
            return True
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
1) Document passages under [SOURCE n] - Text: lines only.
2) Web snippets under [WEB n] - Snippet: lines only.

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
    return _remove_emdashes(_coerce_text_content(response.content).strip()) or "No answer generated."


def generate_blended_answer(
    query: str,
    doc_chunks: list[RetrievedChunk],
    web_context_block: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> GroundedAnswer:
    if not query.strip():
        raise ValueError("Query must be non-empty.")
    doc_block = format_context_for_prompt(doc_chunks, query=query)
    model = create_chat_llm(model=chat_model, temperature=0.0, max_tokens=_GROUNDED_ANSWER_MAX_TOKENS)
    messages = build_blended_messages(query, doc_block, web_context_block)
    response = model.invoke(messages)
    answer = _remove_emdashes(_coerce_text_content(response.content).strip()) or UNKNOWN_PHRASE
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
    section_navigation_query: bool = False,
    broad_document_summary: bool = False,
) -> Iterator[str]:
    """Yield text tokens for Streamlit ``st.write_stream`` (document-grounded only)."""
    if not retrieved_chunks:
        yield "No passages retrieved."
        return
    context_block = format_context_for_prompt(retrieved_chunks, query=query)
    messages = build_grounded_messages(
        query,
        context_block,
        n_context_sources=len(retrieved_chunks),
        section_navigation_query=section_navigation_query,
        broad_document_summary=broad_document_summary,
    )
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


def stream_document_abstain_tokens(
    query: str,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.2,
) -> Iterator[str]:
    if not query.strip():
        yield ""
        return
    model = create_chat_llm(
        model=chat_model,
        temperature=temperature,
        max_tokens=min(220, _GENERAL_ANSWER_MAX_TOKENS),
    )
    messages = [
        SystemMessage(content=DOCUMENT_ABSTAIN_GENERAL_PROMPT),
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
    doc_block = format_context_for_prompt(doc_chunks, query=query)
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
    section_navigation_query: bool = False,
    broad_document_summary: bool = False,
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

    context_block = format_context_for_prompt(retrieved_chunks, query=query)
    messages = build_grounded_messages(
        query,
        context_block,
        n_context_sources=len(retrieved_chunks),
        section_navigation_query=section_navigation_query,
        broad_document_summary=broad_document_summary,
    )
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
