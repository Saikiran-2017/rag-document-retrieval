"""
Centralized query normalization for routing, intent, retrieval, and extraction.

Use the original user string for UI and persistence. Pass the return value of
:func:`normalize_query_for_pipeline` into all logic that previously consumed the
raw message (intent, rewrite, hints, deterministic helpers).

Conservative: explicit typo and phrase maps plus a small set of regex token fixes.
No LLM. Does not invent entities or change semantics beyond obvious misspellings
and light paraphrase for intent stability.
"""

from __future__ import annotations

import re


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# (pattern, replacement) — applied in order with case-insensitive matching.
_PHRASE_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    (r"(?i)\bsummarize\s+this\s*$", "summarize this document"),
    (r"\bsummarise\b", "summarize"),
    (r"\bsumarize\b", "summarize"),
    # Support "what is this/document/file about" as summary variant
    (r"(?i)\b(what|help me understand)\s+(?:is\s+)?(?:this|the)(?:\s+document|\s+file)?\s+about\b", "summarize this document"),
    (r"\bsumary\b", "summary"),
    (r"\bsmmary\b", "summary"),
    (r"\bexplian\b", "explain"),
    (r"\bcontact\s+email\b", "email"),
    (r"\bcontact\s+number\b", "phone number"),
    (r"\bcompany\s+site\b", "website"),
    (r"\bfile\s*name\b", "document name"),
    (r"\bfilename\b", "document name"),
    (r"\bprojects\s+mentioned\b", "projects"),
    (r"\bprograms\s+mentioned\b", "programs"),
    (r"\btechnologies\s+are\s+used\b", "technologies"),
    (r"\btechnologies\s+used\b", "technologies"),
    (r"\bwhat\s+company\s+is\s+this\s+about\b", "what company is discussed"),
    (r"\bsummarize\s+this\s+doc\b", "summarize this document"),
    (r"\bexplain\s+this\s+doc\b", "explain this document"),
    (r"\bsummarize\s+this\s+file\b", "summarize this document"),
    (r"\bdoe\s+he\b", "does he"),
    (r"\bdoe\s+she\b", "does she"),
    (r"\bdoe\s+they\b", "do they"),
    (r"\bdid\s+they\s+included\b", "did they include"),
    (r"\bfullname\b", "full name"),
)

# Substring fixes on lowered text.
_TOKEN_TYPOS: tuple[tuple[str, str], ...] = (
    ("webiste", "website"),
    ("conatct", "contact"),
    ("adress", "address"),
    ("emial", "email"),
    ("phne", "phone"),
    ("nuber", "number"),
    ("numbr", "number"),
    ("fone", "phone"),
    ("proejcts", "projects"),
    ("projets", "projects"),
    ("technolgies", "technologies"),
)

# Whole-token near-misses for high-value intent words (conservative).
_FUZZY_TOKEN_FIXES: tuple[tuple[str, str], ...] = (
    (r"(?i)\bsumm?ar(i|e)ze\b", "summarize"),
    (r"(?i)\bsummarzie\b", "summarize"),
    (r"(?i)\bsummarizee\b", "summarize"),
    (r"(?i)\bsummariy\b", "summary"),
    (r"(?i)\bsummery\b", "summary"),
    (r"(?i)\bexplan\b", "explain"),
)


def normalize_query_for_pipeline(query: str) -> str:
    """
    Return a single line suitable for intent, retrieval seed, and extraction logic.

    Idempotent for typical inputs (safe to call more than once).
    """
    q = _collapse_ws(query)
    if not q:
        return q
    low = q.lower()
    for pat, rep in _PHRASE_SUBSTITUTIONS:
        low = re.sub(pat, rep, low)
    for bad, good in _TOKEN_TYPOS:
        if bad in low:
            low = low.replace(bad, good)
    q = low
    for pat, rep in _FUZZY_TOKEN_FIXES:
        q = re.sub(pat, rep, q)
    return _collapse_ws(q)
