"""Heuristics for when users expect document retrieval (avoid false general fast paths)."""

from __future__ import annotations

import re

from app.llm.query_normalize import normalize_query_for_pipeline

# If any pattern matches, do not use the no-retrieval fast path.
_DOCUMENT_SCOPE = re.compile(
    r"\b("
    r"document|documents|file|files|pdf|upload|uploaded|page|pages|passage|passages|"
    r"excerpt|excerpts|cite|citation|source\b|sources\b|section|library|chunk|table\b|tables\b|"
    r"these files|my files|the text|summarize|summary|summarise|"
    r"key points?|main points?|bullet points?|takeaways?|highlights?|"
    r"main topics?|key topics?|topics?\b|themes?\b|outline|gist|overview|"
    r"big picture|high[-\s]?level|tl;?dr|"
    r"playbook|internal|handbook|memo|flash|brief|"
    r"revenue|earnings|profit|loss|quarter|q[1-4]\b|fiscal|ceo|cfo|corp|"
    r"finance|financial|acme|"
    r"extract|according to|from the|in the document|this (file|doc|pdf|paper)|"
    r"what does (this|the|it|my)\s+(\w+\s+){0,4}(file|document|playbook)\s+"
    r"(say|discuss|cover|mention)|"
    r"what (is|are)\s+(this|the|it|they)\s+(\w+\s+){0,3}about|"
    r"what (is|are)\s+the\s+(main|key)\s+|"
    r"purpose\s+of\s+(this|the|my)|what\s+is\s+the\s+purpose|"
    r"central\s+theme|thesis|argument|conclusion|introduction|"
    r"explain\s+(this|the)\s+(document|file)|"
    r"describe\s+(this|the)\s+(document|file)|"
    r"performance|latency|latencies|throughput|benchmark|metric|metrics|"
    r"sla\b|requirements?|p99|discussed|discussion|plain language"
    r"|loan|loans|applicant|application\s+number|disbursed\s+amount|repayment\s+schedule|interest\s+certificate|rate\s+of\s+interest"
    r"|what\s+company|which\s+company|company\s+discussed|organization\s+discussed|employer\s+discussed"
    r"|what\s+projects|projects?\s+mentioned|programs?\s+mentioned|initiatives?\s+mentioned"
    r"|what\s+technologies|technologies\s+(used|does|do|are)|technology\s+stack"
    r"|how\s+many\s+employees|employee\s+count|headcount|workforce\s+size"
    r"|website\b|web\s*site|homepage|contact\s+email|contact\s+number"
    r"|\bdoes\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+know\b"
    r")\b",
    re.I,
)

# Obvious structured-field asks (email/phone/address/name) must not take the no-retrieval general fast path.
_STRUCTURED_FIELD_DOC_Q = re.compile(
    r"("
    r"\bwhat\s*(?:'s|is|was)\s+(?:the\s+)?(?:his|her|their|its|my|your|our)?\s*"
    r"(?:e-?mail(?:\s+address)?|email(?:\s+address)?|contact\s+e-?mail|contact\s+email|"
    r"phone(?:\s+number)?|contact(?:\s+number)?|mobile\b|"
    r"(?:full\s+)?name\b|(?:current\s+)?address\b|website\b|web\s*page|homepage|\burl\b)\b"
    r"|"
    r"\b(?:his|her|their|its|my|your)\s+(?:e-?mail|email|phone|mobile|address|name)\b"
    r"|"
    r"\b(?:e-?mail|email|phone|contact|address)\s+(?:on\s+file|in\s+the\s+(?:file|document))\b"
    r")",
    re.I,
)

# Assistant / product identity — never treat as document-scoped (Phase M).
_ASSISTANT_IDENTITY = re.compile(
    r"^\s*("
    r"who\s+are\s+you\b|"
    r"what\s+are\s+you\b|"
    r"what(?:'s|\s+is)\s+your\s+name\b|"
    r"who\s+(built|created|made|owns|runs)\s+(you|this(\s+app|\s+assistant|\s+tool|\s+product|\s+service)?)\b|"
    r"what\s+are\s+you\s+called\b|"
    r"what\s+model\s+are\s+you\b|"
    r"what\s+is\s+this\s+(app|tool|assistant|product|service|website)\b"
    r")\s*[?.!]*\s*$",
    re.I,
)

# Short general knowledge role / concept prompts (Phase N) — not document questions by themselves.
_GENERAL_ROLE_OR_CONCEPT = re.compile(
    r"^\s*("
    r"(what\s+is\s+)?(an?\s+)?(ml|machine\s+learning)\s+engineer\b|"
    r"(what\s+is\s+)?(a\s+)?data\s+(engineer|scientist)\b|"
    r"(what\s+is\s+)?(a\s+)?machine\s+learning\s+engineer\b|"
    r"(what\s+is\s+)?(an?\s+)?ml\b(?!\s+engineer)\b|"
    r"(what\s+is\s+)?(a\s+)?machine\s+learning\b|"
    r"machine\s+learning\b|"
    r"data\s+(engineer|scientist)\b|"
    r"(ml|machine\s+learning)\s+engineer\b"
    r")\s*[?.!]*\s*$",
    re.I,
)


def is_assistant_identity_question(query: str) -> bool:
    """True when the user asks about the assistant or host product, not library content."""
    q = (query or "").strip()
    if len(q) < 3:
        return False
    return bool(_ASSISTANT_IDENTITY.search(q))


def is_general_short_concept_query(query: str) -> bool:
    """True for compact definitions of common roles without document deixis."""
    q = (query or "").strip()
    if not q or len(q) > 120:
        return False
    return bool(_GENERAL_ROLE_OR_CONCEPT.search(q))


def should_bypass_document_intent_for_query(query: str) -> bool:
    """True when routing must not treat the turn as document-scoped (identity / general concepts)."""
    q = normalize_query_for_field_intent((query or "").strip())
    return is_assistant_identity_question(q) or is_general_short_concept_query(q)


def normalize_query_for_field_intent(query: str) -> str:
    """Backward-compatible alias for :func:`normalize_query_for_pipeline`."""
    return normalize_query_for_pipeline(query)


# Broad / document-level questions: skip LLM query rewrite and optionally run a second retrieval pass.
_SUMMARY_STANDALONE = re.compile(r"^\s*summarize\s*[?.!]*\s*$", re.I)

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
    r"give\s+me\s+an?\s+overview|brief\s+overview|"
    r"give\s+(me\s+)?(a\s+)?summary\b|"
    r"explain\s+(this|the|my)\s+(document|file)\b|"
    r"summar(?:y|ise|izing)\s+(?:of\s+)?(?:this|the|my)\s+(?:document|file)\b|"
    r"(?:give\s+me\s+)?a?\s*summary\s+of\s+(?:this|the|my)\s+(?:document|file)\b"
    r")\b",
    re.I,
)


def user_expects_document_grounding(query: str) -> bool:
    """True when the question should run retrieval if a library exists (disables general fast path)."""
    q = normalize_query_for_field_intent((query or "").strip())
    if len(q) < 4:
        return False
    if should_bypass_document_intent_for_query(q):
        return False
    if len(q) <= 160 and _STRUCTURED_FIELD_DOC_Q.search(q):
        return True
    return bool(_DOCUMENT_SCOPE.search(q))


def is_broad_document_overview_query(query: str) -> bool:
    """True for high-level doc questions that need diverse chunks and should not be over-compressed by rewrite."""
    q = normalize_query_for_field_intent((query or "").strip())
    if len(q) < 4:
        return False
    if _SUMMARY_STANDALONE.match(q):
        return True
    if len(q) < 6:
        return False
    return bool(_BROAD_OVERVIEW.search(q))


def uses_relaxed_document_grounding_gate(query: str) -> bool:
    """
    Use looser hybrid + ready_limited gates (still require non-empty trusted hits).

    Covers broad/summary-style questions and vague \"how is X discussed\" doc queries
    where top-hit L2 is often higher but RRF / multi-chunk support is still strong.

    Excludes narrow fact lookups (handled by strict gate) and does not key off
    negative/absurd topics.
    """
    if is_broad_document_overview_query(query):
        return True
    q = normalize_query_for_field_intent((query or "").strip()).lower()
    if len(q) < 10:
        return False
    if re.search(
        r"\b(what\s+company|which\s+company|company\s+discussed|organization\s+discussed)\b",
        q,
    ):
        return True
    if re.search(r"\bwhat\s+projects\b|\bprojects?\s+mentioned\b", q):
        return True
    if re.search(r"\bwhat\s+technologies\b|\btechnologies\s+(does|do|are|used)\b", q):
        return True
    perf = bool(
        re.search(
            r"\b(performance|latency|p99|throughput|workload|sla|benchmark|metrics?|reliability)\b",
            q,
        )
    )
    vague = bool(
        re.search(
            r"\b(how|what)\b.+\b(discussed|discussion|covered|addressed|talked\s+about)\b",
            q,
        )
    )
    return perf and vague


# Narrow entity / role questions: relax hybrid L2 slightly when RRF is strong (Phase 29).
_ENTITY_LOOKUP_SAFE = re.compile(
    r"\b("
    r"company\b|organization\b|employer\b|"
    r"website\b|homepage\b|\burl\b|"
    r"named\s+as\b|"
    r"who\s+is\s+named\b|"
    r"name\s+of\b|"
    r"who\s+is\b|who\s+was\b|"
    r"owner\b|maintainer\b|on-?call\b|"
    r"manager\b|director\b|lead\b|"
    r"author\b|reviewer\b|approver\b|"
    r"contact\b|point\s+of\s+contact\b|"
    r"cfo\b|ceo\b|cto\b|coo\b|ciso\b|"
    r"finance\b|financial\b|accounting\b|"
    r"effective\s+date\b|expires?\b|deadline\b|due\s+date\b|"
    r"invoice\b|po\b|purchase\s+order\b|order\s+id\b|"
    r"ticket\b|case\b|issue\b|incident\b|"
    r"identifier\b|id\b|uuid\b|"
    r"serial\b|tracking\b|reference\b|ref\b"
    r"|applicant\b|application\s+number\b|disbursed\s+amount\b|loan\s+amount\b"
    r"|e-?mail\b|email\b|phone\b|mobile\b|address\b|postal\b|zip\b"
    r")\b",
    re.I,
)

_LOOKUP_NEGATIVE = re.compile(
    r"\b(recipe|chocolate cake|land on mars|on mars|exact recipe|impossible fact)\b",
    re.I,
)


def is_sparse_entity_lookup_query(query: str) -> bool:
    """
    True for short entity/role lookups where vector distance is often just above strict QA
    but hybrid RRF still shows a good keyword match (e.g. CFO name).
    """
    q = normalize_query_for_pipeline((query or "").strip())
    if len(q) < 10:
        return False
    if _LOOKUP_NEGATIVE.search(q):
        return False
    ql = q.lower()
    # Fast shape heuristic: short WH- questions or "name/id/date" asks.
    wh = bool(re.match(r"^\s*(who|what|when)\b", ql))
    ask = bool(
        re.search(
            r"\b(name|named|id|identifier|uuid|date|owner|lead|manager|contact|applicant|application|loan|disbursed|"
            r"e-?mail|email|phone|mobile|address|website|homepage|url)\b",
            ql,
        )
    )
    if not (wh or ask):
        return False
    return bool(_ENTITY_LOOKUP_SAFE.search(q))


def is_section_navigation_query(query: str) -> bool:
    """
    True when the user targets a structural part of a document (section N, DR topic, appendix).

    Used to widen retrieval, boost heading-like matches, and pass more diverse chunks to the LLM.
    """
    q = normalize_query_for_pipeline((query or "").strip()).lower()
    if len(q) < 12:
        return False
    if re.search(r"\bsection\s*(?:#|number|num\.?)?\s*\d+\b", q):
        return True
    if "section" in q and re.search(r"\b(seven|eight|nine|ten|eleven|twelve|\d+)\b", q):
        return True
    # Common phrasing without the literal word "section"
    if re.search(r"\b(part|chapter|appendix|annex)\s+\d+\b", q):
        return True
    if re.search(r"\b(in|under)\s+(appendix|chapter)\b", q):
        return True
    if "disaster" in q and "recover" in q:
        return True
    if re.search(r"\bwhat\s+.*\b(section|subsection|appendix)\b", q):
        return True
    if re.search(r"\bchapter\s+\d+\b", q):
        return True
    return False
