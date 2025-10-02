"""
Lightweight checks that grounded answers stay within supplied sources.

No extra LLM calls: citation parsing + token overlap vs context.
"""

from __future__ import annotations

import re
from typing import Any

from app.retrieval.vector_store import RetrievedChunk
from app.reliability.turn_log import log_validation_failure
from app.services.web_search_service import normalize_web_identity

# Minimum share of non-trivial answer tokens that appear in context (when answer is substantive).
_MIN_LEXICAL_OVERLAP = 0.12

_STOP = frozenset(
    "a an the and or but if to of in on for with as at by from is are was were be been being "
    "it this that these those i you we they he she not no yes so than then there here do does did "
    "can could should would will just only very more most also into about out up what which who "
    "how when where why all any each both few such than too very".split()
)


def _tokenize_loose(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if t not in _STOP
    }


def citations_in_answer(answer: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"\[SOURCE\s+(\d+)\s*\]", answer, re.I)]


def validate_source_citations(answer: str, n_sources: int) -> bool:
    """Every [SOURCE n] must satisfy 1 <= n <= n_sources."""
    if n_sources <= 0:
        return True
    nums = citations_in_answer(answer)
    if not nums:
        return True
    return all(1 <= n <= n_sources for n in nums)


def lexical_support_ratio(answer: str, chunks: list[RetrievedChunk]) -> float:
    """Share of answer content tokens that appear in any chunk body."""
    if not answer.strip() or not chunks:
        return 1.0
    a_tok = _tokenize_loose(answer)
    if len(a_tok) < 6:
        return 1.0
    corpus = " ".join((c.page_content or "") for c in chunks).lower()
    corpus_tok = _tokenize_loose(corpus)
    if not corpus_tok:
        return 0.0
    hit = sum(1 for t in a_tok if t in corpus_tok)
    return hit / max(len(a_tok), 1)


def validate_grounded_answer(
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    unknown_phrase: str,
) -> tuple[str, str | None]:
    """
    Return ``(answer_text, optional_warning)``.

    - Invalid [SOURCE n] → replace with ``unknown_phrase`` (prevents fake citations).
    - Low lexical overlap on substantive answers → short warning for UI note.
    """
    n = len(chunks)
    stripped = (answer or "").strip()
    if not stripped:
        return unknown_phrase, None
    if unknown_phrase.lower() in stripped.lower():
        return stripped, None

    if not validate_source_citations(stripped, n):
        log_validation_failure("invalid_source_citation", n_sources=n, answer_len=len(stripped))
        return unknown_phrase, "Answer cited a source number not in this reply's context; reply was withheld."

    ratio = lexical_support_ratio(stripped, chunks)
    if len(stripped) > 100 and ratio < _MIN_LEXICAL_OVERLAP:
        return (
            stripped,
            "Automated check: much of the wording may not appear in the retrieved excerpts; verify in Sources.",
        )
    return stripped, None


_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def validate_web_markdown_links(answer: str, allowed_raw_urls: list[str]) -> tuple[str, str | None]:
    """
    Strip or unlink markdown URLs that are not from the retrieved web result set.

    Returns ``(answer_text, optional_warning)`` for UI notes.
    """
    stripped = (answer or "").strip()
    if not stripped or not allowed_raw_urls:
        return answer, None
    allowed = {u for u in (normalize_web_identity(x) for x in allowed_raw_urls) if u}
    if not allowed:
        return answer, None
    removed = False

    def repl(m: re.Match[str]) -> str:
        nonlocal removed
        text, url = m.group(1), m.group(2)
        nid = normalize_web_identity(url)
        if nid and nid in allowed:
            return m.group(0)
        removed = True
        return (text or "").strip() or url

    fixed = _MD_LINK.sub(repl, stripped)
    warn = (
        "Some web links did not match retrieved results and were removed from the answer."
        if removed
        else None
    )
    if removed:
        log_validation_failure("web_markdown_link_removed", allowed_url_count=len(allowed_raw_urls))
    return fixed, warn
