"""
Conversation-aware retrieval hints for document chat.

Follow-up questions often omit entities ("what is his name", "in this file").
The HTTP API historically sent only the latest message, so retrieval saw a vague query.

This module derives a retrieval query string and optional focus source from prior turns.
It is conservative: only activates after a grounded assistant reply with identifiable sources.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.llm.query_normalize import normalize_query_for_pipeline


@dataclass(frozen=True)
class ConversationRetrievalHints:
    """Augments the current user message for hybrid search and context selection."""

    retrieval_query: str
    focus_source_name: str | None
    force_document_scoped_routing: bool
    relax_lookup_gate: bool


_FOLLOW_PRONOUN = re.compile(
    r"\b(his|her|their|its)\s+(name|full\s*name|title|role|job|position|skills?|experience|background|"
    r"address|email|e-?mail|phone|mobile|number)\b",
    re.I,
)
_FOLLOW_SUBJECT_AUX = re.compile(
    r"\b(does|did|do|is|are|was|were|can|could|would|will)\s+[\w']{0,22}?\b(he|she|they|him|her)\b"
    r"|\b(he|she|they)\s+(know|knows|knew|knw|knws|have|has|had|work|works|worked|use|uses|used)\b"
    r"|\b(doe|does|did)\s+he\b"
    r"|\b(where|when)\s+(does|do|did|is|was)\s+[\w']{0,16}?\b(he|she|they)\b",
    re.I,
)
_FOLLOW_DOC_DEICTIC = re.compile(
    r"\b(in|from|on)\s+(the|this|my|that)\s+(document|file|upload|pdf|uploaded\s+file|material)\b"
    r"|\b(in|from)\s+document\b"
    r"|\b(in|from)\s+the\s+file\b",
    re.I,
)
_FOLLOW_CONFIRM = re.compile(r"^\s*(yes|yeah|yep|yup|ok|okay|sure)\s*[!.?]*\s*$", re.I)
_FOLLOW_BARE_FIELD = re.compile(
    r"^\s*("
    r"phone(\s+number)?|contact(\s+number)?|mobile|cell|"
    r"email|e-?mail(\s+address)?|address|website|url|"
    r"headquarters|headquartered|hq|founded|established|incorporated"
    r")\s*[!.?]*\s*$",
    re.I,
)
_FOLLOW_WH_START = re.compile(
    r"^\s*(what|who|when|where|which|how|did|does|do|is|are|was|were|can|could|would)\b",
    re.I,
)
_METADATA_Q = re.compile(
    r"\b("
    r"file\s*name|filename|document\s*name|name\s+of\s+(the\s+)?(file|document)|"
    r"which\s+file|what\s+file|what\s+is\s+the\s+(file|document)|"
    r"which\s+document(\s+is\s+this)?|what\s+document(\s+is\s+this)?|"
    r"source\s+file"
    r")\b",
    re.I,
)
_ADDRESS_Q = re.compile(
    r"\b(address|street|zip|postal|location)\b",
    re.I,
)
_GENERAL_TECH = re.compile(
    r"^\s*what\s+is\s+("
    r"rag\b|retrieval[-\s]?augmented\s+generation|"
    r"embeddings?\b|vector\s+database|vectordb|llm\b|large\s+language\s+model"
    r")\s*\??\s*$",
    re.I,
)

# Short "What is Starship?" style follow-ups after a grounded turn: stay in-document unless
# this is clearly a general definition prompt (RAG, LLM, etc.).
_FOLLOWUP_WHATIS_TERM = re.compile(
    r"^\s*what\s+is\s+([A-Za-z][A-Za-z0-9\-]{1,32})\s*\??\s*$",
    re.I,
)
_FOLLOWUP_WHATIS_BLOCKLIST = frozenset(
    {
        "rag",
        "ml",
        "llm",
        "nlp",
        "sql",
        "api",
        "aws",
        "etl",
        "gpu",
        "cpu",
        "cdn",
        "http",
        "https",
        "json",
        "xml",
        "csv",
        "pdf",
        "gpt",
    }
)


def _extract_source_names_from_assistant_message(msg: dict[str, Any]) -> list[str]:
    raw = msg.get("sources")
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            sn = str(item.get("source_name") or "").strip()
            if sn:
                names.append(sn)
    return names


def _is_grounded_assistant(msg: dict[str, Any]) -> bool:
    if str(msg.get("role") or "") != "assistant":
        return False
    if msg.get("grounded") is True:
        return True
    if msg.get("sources"):
        return True
    mode = str(msg.get("mode") or "").lower()
    return mode in ("grounded", "blended")


def _prior_user_before_assistant(history: list[dict[str, Any]], asst_idx: int) -> str | None:
    if asst_idx <= 0:
        return None
    prev = history[asst_idx - 1]
    if str(prev.get("role") or "") != "user":
        return None
    t = str(prev.get("content") or "").strip()
    return t or None


def _find_last_grounded_assistant(history: list[dict[str, Any]]) -> tuple[int, list[str]] | None:
    for i in range(len(history) - 1, -1, -1):
        msg = history[i]
        if not _is_grounded_assistant(msg):
            continue
        srcs = _extract_source_names_from_assistant_message(msg)
        if not srcs:
            continue
        return i, srcs
    return None


def _dominant_name(names: list[str]) -> str | None:
    if not names:
        return None
    top, n = Counter(names).most_common(1)[0]
    if not str(top).strip():
        return None
    if n == 1 and len(set(names)) > 3:
        # Too fragmented; avoid pinning to a single file.
        return None
    return str(top).strip()


def _looks_like_doc_followup(query: str) -> bool:
    q = normalize_query_for_pipeline((query or "").strip())
    if not q:
        return False
    from app.llm.query_intent import should_bypass_document_intent_for_query

    if should_bypass_document_intent_for_query(q):
        return False
    if _GENERAL_TECH.search(q):
        return False
    # Expansion queries: tell me more, continue, details, elaborate, etc.
    if re.search(
        r"^\s*(tell\s+me\s+more|continue|details|elaborate|more\s+details|expand|what\s+else|more|further)\s*[!.?]*\s*$",
        q,
        re.I,
    ):
        return True
    if _FOLLOW_PRONOUN.search(q):
        return True
    if _FOLLOW_SUBJECT_AUX.search(q):
        return True
    if _FOLLOW_DOC_DEICTIC.search(q):
        return True
    # Ultra-short recoveries after a grounded turn: treat as doc-scoped only when
    # conversation context already provides a dominant library source.
    if _FOLLOW_CONFIRM.search(q):
        return True
    if _FOLLOW_BARE_FIELD.search(q):
        return True
    if _METADATA_Q.search(q):
        return True
    if _ADDRESS_Q.search(q) and len(q) < 160:
        return True
    if re.search(r"\b(the\s+same|as\s+above|you\s+just\s+said|previous\s+answer)\b", q, re.I):
        return True
    if re.search(r"\b(include|included|mention|mentions|list|lists)\b", q, re.I) and len(q) < 180:
        return True
    if re.search(r"\bwhat\s+about\s+(his|her|their|the|this|my|your)\b", q, re.I):
        return True
    m_term = _FOLLOWUP_WHATIS_TERM.match(q)
    if m_term and len(q) <= 72 and not _GENERAL_TECH.search(q):
        term = m_term.group(1).lower()
        if term not in _FOLLOWUP_WHATIS_BLOCKLIST:
            return True
    # Short WH-questions only count as doc follow-ups when they clearly anchor to the library
    # (pronoun / deictic / doc-shaped noun)—not every vague "what is …" after a grounded reply.
    if len(q) <= 100 and _FOLLOW_WH_START.search(q):
        if re.search(r"\b(his|her|their|its|he|she|they|him)\b", q, re.I):
            return True
        if re.search(
            r"\b(the|this|my|that|your)\s+"
            r"(loan|application|file|document|upload|library|amount|balance|name|number|address|"
            r"city|email|phone|record|policy|packet|form|memo|brief|playbook)\b",
            q,
            re.I,
        ):
            return True
        if re.search(
            r"\b(my|your)\s+(loan|file|document|application|upload|policy|library)\b",
            q,
            re.I,
        ):
            return True
    return False


def build_conversation_retrieval_hints(
    query: str,
    conversation_history: list[dict[str, Any]] | None,
) -> ConversationRetrievalHints:
    """
    ``conversation_history`` should be turns *before* the current user message, oldest-first.
    Each item may include standard chat fields plus optional ``sources`` / ``mode`` from SQLite extras.
    """
    q = normalize_query_for_pipeline((query or "").strip())
    hist = [m for m in (conversation_history or []) if isinstance(m, dict)]
    base = ConversationRetrievalHints(
        retrieval_query=q,
        focus_source_name=None,
        force_document_scoped_routing=False,
        relax_lookup_gate=False,
    )
    if not q or not hist:
        return base

    grounded = _find_last_grounded_assistant(hist)
    if grounded is None:
        return base
    asst_idx, src_names = grounded
    dominant = _dominant_name(src_names)
    if not dominant:
        return base

    from app.llm.query_intent import should_bypass_document_intent_for_query

    if should_bypass_document_intent_for_query(q):
        return base

    if not _looks_like_doc_followup(q):
        return base

    prev_user = _prior_user_before_assistant(hist, asst_idx)
    parts: list[str] = []
    if prev_user:
        prev_user = prev_user.strip()
        if len(prev_user) > 260:
            prev_user = prev_user[-260:]
        prev_user = normalize_query_for_pipeline(prev_user)
        parts.append(prev_user)
    parts.append(q)
    parts.append(dominant)
    merged = " ".join(p for p in parts if p).strip()
    if len(merged) > 520:
        merged = merged[:520]

    m_what_is = _FOLLOWUP_WHATIS_TERM.match(q)
    relax = bool(
        _FOLLOW_PRONOUN.search(q)
        or _FOLLOW_SUBJECT_AUX.search(q)
        or _METADATA_Q.search(q)
        or _ADDRESS_Q.search(q)
        or _FOLLOW_DOC_DEICTIC.search(q)
        or (
            m_what_is
            and len(q) <= 72
            and not _GENERAL_TECH.search(q)
            and (m_what_is.group(1).lower() not in _FOLLOWUP_WHATIS_BLOCKLIST)
        )
        or (
            _FOLLOW_WH_START.search(q)
            and len(q) < 100
            and (
                re.search(r"\b(his|her|their|its|he|she|they|him)\b", q, re.I)
                or re.search(
                    r"\b(the|this|my|that|your)\s+"
                    r"(loan|application|file|document|upload|library|amount|balance|name|number|address)\b",
                    q,
                    re.I,
                )
                or re.search(
                    r"\b(my|your)\s+(loan|file|document|application|upload|policy|library)\b",
                    q,
                    re.I,
                )
            )
        )
    )

    return ConversationRetrievalHints(
        retrieval_query=merged,
        focus_source_name=dominant,
        force_document_scoped_routing=True,
        relax_lookup_gate=relax,
    )


def is_short_document_deictic_followup(query: str) -> bool:
    """True for very short turns like ``in document`` that anchor to prior grounded context."""
    q = normalize_query_for_pipeline((query or "").strip())
    if len(q) > 88:
        return False
    return bool(_FOLLOW_DOC_DEICTIC.search(q))


def effective_user_expects_document_grounding(query: str, hints: ConversationRetrievalHints) -> bool:
    from app.llm.query_intent import (
        should_bypass_document_intent_for_query,
        user_expects_document_grounding,
    )

    if should_bypass_document_intent_for_query(query):
        return False
    return user_expects_document_grounding(query) or hints.force_document_scoped_routing
