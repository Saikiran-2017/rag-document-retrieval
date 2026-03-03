"""Heuristics for when users expect document retrieval (avoid false general fast paths)."""

from __future__ import annotations

import re

# If any pattern matches, do not use the no-retrieval fast path.
_DOCUMENT_SCOPE = re.compile(
    r"\b("
    r"document|documents|file|files|pdf|upload|uploaded|page|pages|passage|passages|"
    r"excerpt|excerpts|cite|citation|source\b|sources\b|section|library|chunk|table\b|tables\b|"
    r"these files|my files|the text|summarize|summary|summarise|"
    r"key points?|main points?|bullet points?|takeaways?|highlights?|"
    r"main topics?|key topics?|topics?\b|themes?\b|outline|gist|overview|"
    r"big picture|high[-\s]?level|tl;?dr|"
    r"extract|according to|from the|in the document|this (file|doc|pdf|paper)|"
    r"what does (this|the|it)\s+(\w+\s+)?(file|document)\s+discuss|"
    r"what (is|are)\s+(this|the|it|they)\s+about|"
    r"what (is|are)\s+the\s+(main|key)\s+|"
    r"purpose\s+of\s+(this|the|my)|what\s+is\s+the\s+purpose|"
    r"central\s+theme|thesis|argument|conclusion|introduction|"
    r"explain\s+(this|the)\s+(document|file)|"
    r"describe\s+(this|the)\s+(document|file)|"
    r"performance|latency|latencies|throughput|benchmark|metric|metrics|"
    r"sla\b|requirements?|p99|discussed|discussion"
    r")\b",
    re.I,
)

# Broad / document-level questions: skip LLM query rewrite and optionally run a second retrieval pass.
_BROAD_OVERVIEW = re.compile(
    r"\b("
    r"what\s+is\s+(this|the|it)\s+.+\s+about|"
    r"what\s+(is|are)\s+(this|the|it)\s+about|"
    r"summarize|summarise|summary\b|overview\b|synopsis\b|"
    r"main\s+points?|key\s+points?|key\s+ideas?|main\s+ideas?|"
    r"main\s+topics?|key\s+topics?|"
    r"what\s+are\s+the\s+(main|key)\s+(points?|topics?|themes?|ideas?)|"
    r"what\s+does\s+(this|the)\s+(file|document)\s+discuss|"
    r"what\s+is\s+the\s+(purpose|topic|subject|gist|main\s+idea)|"
    r"purpose\s+of\s+(this|the|my)\s+(file|document)?|"
    r"high[-\s]?level|big\s+picture|"
    r"central\s+theme|takeaways?|highlights?|"
    r"give\s+me\s+an?\s+overview|brief\s+overview"
    r")\b",
    re.I,
)


def user_expects_document_grounding(query: str) -> bool:
    """True when the question should run retrieval if a library exists (disables general fast path)."""
    q = (query or "").strip()
    if len(q) < 4:
        return False
    return bool(_DOCUMENT_SCOPE.search(q))


def is_broad_document_overview_query(query: str) -> bool:
    """True for high-level doc questions that need diverse chunks and should not be over-compressed by rewrite."""
    q = (query or "").strip()
    if len(q) < 6:
        return False
    return bool(_BROAD_OVERVIEW.search(q))
