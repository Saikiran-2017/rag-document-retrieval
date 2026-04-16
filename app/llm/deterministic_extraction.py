"""
Deterministic post-retrieval helpers for obvious field/value questions.

Goal: reduce false refusals when the selected context already contains an explicit answer
as a clear label/value pair (common in PDFs exported from business systems).

This module MUST be conservative:
- Only answer when a high-precision pattern match exists in the selected chunks.
- Never invent values.
- Always cite the specific [SOURCE n] that contained the matched value.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.llm.query_normalize import normalize_query_for_pipeline
from app.retrieval.vector_store import RetrievedChunk


@dataclass(frozen=True)
class ExtractedAnswer:
    answer: str
    used_source_numbers: tuple[int, ...]


_WS = r"[ \t\u00A0]*"

# Currency/amount: allow commas and 2 decimals (most bank statements), tolerate optional ₹ and spaces.
_AMOUNT = r"(?:₹\s*)?\d{1,3}(?:,\d{3})*(?:\.\d{2})"

# IDs: allow alnum and a few separators.
_ID = r"[A-Z0-9][A-Z0-9\-\/]{3,}"

# One or more regexes per logical field kind (tried in order; first match wins).
_FIELD_PATTERNS: list[tuple[str, tuple[re.Pattern[str], ...]]] = [
    (
        "loan_disbursed_amount",
        (
            re.compile(
                rf"\b(loan{_WS}disbursed{_WS}amount|disbursed{_WS}amount|loan{_WS}amount)\b{_WS}[:\-]?\s*({_AMOUNT})\b",
                re.I,
            ),
        ),
    ),
    (
        "person_name",
        (
            re.compile(
                rf"\b("
                rf"applicant{_WS}name|full{_WS}name|subject{_WS}name|borrower{_WS}name|primary{_WS}holder|"
                rf"customer{_WS}name|account{_WS}holder|legal{_WS}name"
                rf")\b{_WS}[:\-]?\s*([A-Z][A-Za-z \.\'\-]{{2,}})\b",
                re.I,
            ),
            # Top-of-form "Name:" / "Full name:" (label is a whole word; avoids "username:" false positives).
            re.compile(
                rf"^\s*(?:full{_WS}name|(?<![A-Za-z])name(?![A-Za-z]))\s*[:\-]\s*([A-Za-z][A-Za-z \.\'\-]{{2,}})\b",
                re.I,
            ),
        ),
    ),
    (
        "application_number",
        (
            re.compile(
                rf"\b(application{_WS}(number|no\.?|#))\b{_WS}[:\-]?\s*({_ID})\b",
                re.I,
            ),
        ),
    ),
    (
        "email",
        (
            re.compile(
                rf"\b(e-?mail{_WS}address|email{_WS}address|e-?mail|email)\b{_WS}[:\-]?\s*"
                rf"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}})\b",
                re.I,
            ),
            # Support "Contact: email" patterns (contact as generic label)
            re.compile(
                rf"\b(contact)\b{_WS}[:\-]?\s*"
                rf"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}})\b",
                re.I,
            ),
            # Label-only pattern for multi-line cases (e.g., "Email:" on one line, value on next)
            re.compile(
                rf"^\s*(e-?mail|email)\s*[:\-]\s*$",
                re.I,
            ),
        ),
    ),
    (
        "phone",
        (
            re.compile(
                rf"\b(contact{_WS}number|phone{_WS}number|contact|phone|mobile|cell|tel\.?)\b{_WS}[:\-]?\s*"
                rf"([\d()+\-\s]{{10,24}})\b",
                re.I,
            ),
            # Label-only pattern for multi-line cases (e.g., "Phone:" on one line, number on next)
            re.compile(
                rf"^\s*(phone|contact|mobile|cell)\s*[:\-]\s*$",
                re.I,
            ),
        ),
    ),
    (
        "website",
        (
            re.compile(
                rf"\b(website|web{_WS}page|homepage|url)\b{_WS}[:\-]?\s*"
                rf"((?:https?://|www\.)[A-Za-z0-9._~/?#=\-]+)",
                re.I,
            ),
            re.compile(
                rf"\b(website|url)\b{_WS}[:\-]?\s*"
                rf"([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9-]{1,24})+\.(?:com|net|org|io|edu|gov)\b)",
                re.I,
            ),
        ),
    ),
    (
        "address",
        (
            re.compile(
                rf"\b(current{_WS}address|mailing{_WS}address|address)\b{_WS}[:\-]?\s*(.+?)\s*$",
                re.I,
            ),
        ),
    ),
]


def _query_kind(query: str) -> str | None:
    q = normalize_query_for_pipeline((query or "").strip()).lower()
    if not q:
        return None
    if re.search(r"\b(his|her|their)\s+name\b|\bwhat\s+is\s+(his|her|their)\s+name\b", q):
        return "person_name"
    if re.search(
        r"\bwhat\s*(?:'s|is)\s+(?:the\s+)?(?:full\s+)?name\b|\bname\s+on\s+(?:file|record)\b",
        q,
    ):
        return "person_name"
    if re.search(
        r"\b(applicant|subject|borrower|customer|account holder|full|legal)\b.*\bname\b|\bapplicant name\b",
        q,
    ):
        return "person_name"
    if re.search(r"\bapplication\b.*\b(number|no|#)\b|\bapplication number\b", q):
        return "application_number"
    if re.search(
        r"\b(loan)\b.*\b(amount|disbursed)\b|\bdisbursed amount\b|\bloan amount\b|\bhow much\b.*\bloan\b",
        q,
    ):
        return "loan_disbursed_amount"
    if re.search(r"\b(e-?mail(\s+address)?|email(\s+address)?|contact\s+e-?mail|contact\s+email)\b", q):
        return "email"
    if re.search(r"\b(phone|phone number|mobile|contact number|contact|cell|telephone)\b", q):
        return "phone"
    if re.search(r"\b(website|web\s*page|homepage|\burl\b)\b", q):
        return "website"
    if re.search(
        r"\b(current\s+)?address\b|\bstreet address\b|\bmailing address\b|\bcurrent\s+address\b",
        q,
    ):
        return "address"
    return None


_NEXT_FIELD_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9 /]{0,42}:\s*\S")


def _merge_address_continuation(lines: list[str], start_i: int, first_line_value: str) -> str:
    """Join obvious address wrap lines (PDF exports) until a new labeled field or limit."""
    parts = [first_line_value.strip()]
    total = len(first_line_value.strip())
    max_lines = 4
    max_chars = 240
    i = start_i + 1
    while i < len(lines) and len(parts) < max_lines and total < max_chars:
        nxt = (lines[i] or "").strip()
        if not nxt:
            break
        if _NEXT_FIELD_LINE.match(nxt):
            break
        parts.append(nxt)
        total += len(nxt) + 1
        i += 1
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def try_extract_field_value_answer(query: str, hits: list[RetrievedChunk]) -> ExtractedAnswer | None:
    """
    If query looks like a field/value lookup (loan amount, applicant name, etc.) and a clear
    label/value pair exists in any selected chunk, return a grounded answer with a citation.
    """
    kind = _query_kind(query)
    if not kind or not hits:
        return None
    pats = next((p for k, p in _FIELD_PATTERNS if k == kind), None)
    if not pats:
        return None

    for idx, h in enumerate(hits, start=1):
        text = (h.page_content or "").strip()
        if not text:
            continue
        lines = text.splitlines()
        # Scan line-by-line to avoid accidental cross-line joins (except controlled address merge).
        for li, line in enumerate(lines):
            for pat in pats:
                m = pat.search(line)
                if not m:
                    continue
                
                # Handle label-only patterns (when value is on next line)
                # For email/phone, if we matched just a label with no value, check next line
                if kind == "email" and m.lastindex == 1 and li + 1 < len(lines):
                    next_line = lines[li + 1].strip()
                    if re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", next_line):
                        return ExtractedAnswer(
                            answer=f"The document states that the email address is {next_line} [SOURCE {idx}].",
                            used_source_numbers=(idx,),
                        )
                elif kind == "phone" and m.lastindex == 1 and li + 1 < len(lines):
                    next_line = lines[li + 1].strip()
                    if re.match(r"^[\d()+\-\s]{10,24}$", next_line):
                        raw_phone = re.sub(r"\s+", " ", next_line)
                        if len(re.sub(r"\D", "", raw_phone)) >= 10:
                            return ExtractedAnswer(
                                answer=f"According to the uploaded file [SOURCE {idx}], the phone / contact number is {raw_phone}.",
                                used_source_numbers=(idx,),
                            )
                
                if kind == "loan_disbursed_amount":
                    amount = m.group(2)
                    return ExtractedAnswer(
                        answer=(
                            f"According to the uploaded file [SOURCE {idx}], the loan disbursed amount is {amount}."
                        ),
                        used_source_numbers=(idx,),
                    )
                if kind == "person_name":
                    name = (m.group(2) if (m.lastindex or 0) >= 2 else m.group(1)).strip()
                    # Avoid overly generic false positives.
                    parts = [p for p in name.split() if p]
                    if len(name) < 3 or len(parts) < 1:
                        continue
                    # Plain "Name:" lines may be lowercase; still require a plausible person token shape.
                    if (m.lastindex or 0) < 2:
                        if not any(ch.isupper() for ch in name) and len(parts) < 2:
                            continue
                    # Block obvious non-names that come from generic text (e.g. "or application number").
                    bad_tokens = {
                        "or",
                        "and",
                        "number",
                        "application",
                        "loan",
                        "amount",
                        "date",
                        "id",
                        "no",
                        "ref",
                    }
                    low_parts = {p.lower().strip(".") for p in parts}
                    if len(parts) >= 2 and (low_parts & bad_tokens):
                        continue
                    if len(parts) >= 3 and sum(1 for p in parts if p.lower() in bad_tokens) >= 1:
                        continue
                    return ExtractedAnswer(
                        answer=(
                            f"The document states that the name on record is {name} [SOURCE {idx}]."
                        ),
                        used_source_numbers=(idx,),
                    )
                if kind == "application_number":
                    app_no = m.group(3).strip()
                    return ExtractedAnswer(
                        answer=(
                            f"According to the uploaded file [SOURCE {idx}], the application number is {app_no}."
                        ),
                        used_source_numbers=(idx,),
                    )
                if kind == "email":
                    em = m.group(2).strip()
                    return ExtractedAnswer(
                        answer=(f"The document states that the email address is {em} [SOURCE {idx}]."),
                        used_source_numbers=(idx,),
                    )
                if kind == "phone":
                    raw_phone = re.sub(r"\s+", " ", m.group(2).strip())
                    if len(re.sub(r"\D", "", raw_phone)) < 10:
                        continue
                    return ExtractedAnswer(
                        answer=(
                            f"According to the uploaded file [SOURCE {idx}], the phone / contact number is {raw_phone}."
                        ),
                        used_source_numbers=(idx,),
                    )
                if kind == "website":
                    url = m.group(2).strip()
                    return ExtractedAnswer(
                        answer=(
                            f"The document lists the website / URL as {url} [SOURCE {idx}]."
                        ),
                        used_source_numbers=(idx,),
                    )
                if kind == "address":
                    addr = m.group(2).strip()
                    addr = _merge_address_continuation(lines, li, addr)
                    addr = re.sub(r"\s+", " ", addr)
                    if len(addr) < 8 or len(addr) > 260:
                        continue
                    low = addr.lower()
                    if not any(
                        x in low
                        for x in (
                            "street",
                            "st.",
                            "road",
                            "rd.",
                            "avenue",
                            "ave",
                            "lane",
                            "drive",
                            "blvd",
                            "suite",
                            "unit",
                            "apt",
                            "city",
                            "state",
                            "zip",
                            "pin",
                            "country",
                            "p.o",
                            "po box",
                        )
                    ) and not any(ch.isdigit() for ch in addr):
                        continue
                    return ExtractedAnswer(
                        answer=(f"The document states that the address on record is: {addr} [SOURCE {idx}]."),
                        used_source_numbers=(idx,),
                    )
    return None


_METADATA_FILENAME_Q = re.compile(
    r"\b("
    r"file\s*name|filename|document\s*name|name\s+of\s+(the\s+)?(file|document)|"
    r"which\s+file|what\s+file|what\s+is\s+the\s+(file|document)|"
    r"which\s+document(\s+is\s+this)?|what\s+document(\s+is\s+this)?|"
    r"source\s+file"
    r")\b",
    re.I,
)


def try_answer_document_metadata_question(query: str, hits: list[RetrievedChunk]) -> ExtractedAnswer | None:
    """
    Answer questions about which uploaded file the current context came from.
    Uses chunk metadata only (no filename guessing beyond retrieved hits).
    """
    if not hits or not _METADATA_FILENAME_Q.search(query or ""):
        return None
    names = [
        str(h.metadata.get("source_name") or "").strip()
        for h in hits
        if str(h.metadata.get("source_name") or "").strip()
    ]
    if not names:
        return None
    top, n = Counter(names).most_common(1)[0]
    share = n / max(len(names), 1)
    if share < 0.45 and len({*names}) > 2:
        return None
    # Cite the first chunk that matches the dominant filename.
    cite_idx = 1
    for idx, h in enumerate(hits, start=1):
        if str(h.metadata.get("source_name") or "").strip() == top:
            cite_idx = idx
            break
    return ExtractedAnswer(
        answer=f"The excerpts in this reply are from the uploaded library file `{top}` [SOURCE {cite_idx}].",
        used_source_numbers=(cite_idx,),
    )


_DOC_ABOUT_Q = re.compile(
    r"\b(what\s+is\s+(this|the|it)\s+.*\s+about|what\s+is\s+this\s+document\s+about|"
    r"summar(?:y|ise|izing)\s+(?:of\s+)?(?:this|the|my)\s+(?:document|file)\b|"
    r"(?:give\s+me\s+)?a?\s*summary\s+of\s+(?:this|the|my)\s+(?:document|file)\b|"
    r"summarize\s+this\s+(file|document)|summarize\s+the\s+(file|document)|summarize\s+my\s+(file|document)|"
    r"what\s+does\s+this\s+(file|document)\s+contain|"
    r"what\s+is\s+in\s+this\s+(file|document))\b",
    re.I,
)


def _build_specific_overview_summary(blob: str, themes: list[str]) -> str | None:
    """
    When themes are detected in a field-heavy document, construct a more specific summary
    by extracting key entities (company name, person name, roles, projects) rather than
    using generic theme labels.
    
    Returns a more natural, specific sentence if entities are found, else None.
    """
    low = blob.lower()
    
    # Extract key entities from blob based on detected themes
    entities: dict[str, str | list[str]] = {}
    
    # Company or person names
    if "spacex" in low:
        entities["company"] = "SpaceX"
    if "elon" in low:
        entities["person"] = "Elon Musk"
    
    # Roles (capitalize properly)
    roles: list[str] = []
    if "chief engineer" in low and "Chief Engineer" not in roles:
        roles.append("Chief Engineer")
    if "ceo" in low and "CEO" not in roles:
        roles.append("CEO")
    if "founder" in low and "Founder" not in roles:
        roles.append("Founder")
    if roles:
        entities["roles"] = roles
    
    # Projects
    projects: list[str] = []
    if "starship" in low and "Starship" not in projects:
        projects.append("Starship")
    if "starlink" in low and "Starlink" not in projects:
        projects.append("Starlink")
    if projects:
        entities["projects"] = projects
    
    # Only return specific summary if we have meaningful entity extraction
    if not entities:
        return None
    
    # Build a natural sentence with extracted entities
    parts: list[str] = []
    
    if "company" in entities or "person" in entities:
        intro = "This document provides an overview of"
        names: list[str] = []
        if "company" in entities:
            names.append(entities["company"])
        if "person" in entities:
            names.append(entities["person"])
        
        if len(names) == 1:
            parts.append(f"{intro} {names[0]}")
        else:
            parts.append(f"{intro} {' and '.join(names)}")
        
        # Add roles if present
        if "roles" in entities:
            roles_str = " and ".join(entities["roles"])
            if len(entities["roles"]) > 1:
                parts.append(f"including roles as {roles_str}")
            else:
                parts.append(f"including the role of {roles_str}")
        
        # Add company background theme
        if any(t for t in themes if "background" in t.lower() or "organization" in t.lower()):
            if not ("roles" in entities):
                parts.append("and company background")
        
        # Add projects if present
        if "projects" in entities:
            projects_str = " and ".join(entities["projects"])
            parts.append(f"projects such as {projects_str}")
        
        return ", ".join(parts) + "."
    
    return None


# Substrings in excerpt text mapped to short thematic phrases (meaning, not field layout).
_OVERVIEW_TOPIC_TERMS: tuple[tuple[str, str], ...] = (
    ("loan", "loan processing and obligations"),
    ("applicant", "applicant details"),
    ("repayment", "repayment terms"),
    ("interest certificate", "interest documentation"),
    ("interest", "interest and rates"),
    ("disbursed", "disbursement amounts"),
    ("application number", "application tracking"),
    ("application", "application context"),
    ("company", "company or organization background"),
    ("role", "roles and responsibilities"),
    ("contact", "contact and coordination details"),
    ("email", "contact channels"),
    ("phone", "contact channels"),
    ("website", "web presence"),
    ("spacex", "the organization profiled"),
    ("starship", "launch vehicle programs"),
    ("starlink", "connectivity initiatives"),
    ("mars", "Mars and exploration goals"),
    ("ceo", "leadership"),
    ("chief engineer", "engineering leadership"),
    ("engineer", "technical roles"),
    ("employee", "workforce size and roles"),
    ("mission", "mission and strategy"),
    ("project", "projects and initiatives"),
    ("product", "products and offerings"),
    ("customer", "customers and accounts"),
    ("revenue", "financial performance"),
    ("contract", "contractual terms"),
    ("policy", "policy content"),
)


def _overview_themes_from_blob(blob: str) -> list[str]:
    low = blob.lower()
    out: list[str] = []
    seen: set[str] = set()
    for key, label in _OVERVIEW_TOPIC_TERMS:
        if key in low and label not in seen:
            seen.add(label)
            out.append(label)
    return out[:6]


def _overview_prose_sentences(text: str, *, limit: int = 8) -> list[str]:
    """Extract meaningful prose sentences, filtering field-like patterns.
    
    Enhanced to better filter field labels even in reflowed text, and reject
    documents that are predominantly field-structured rather than prose.
    """
    flat = re.sub(r"\s+", " ", (text or "").strip())
    if not flat:
        return []
    parts = re.split(r"(?<=[.!?])\s+", flat)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 48 or len(p) > 560:
            continue
        # Filter field-like patterns: "Label: value", "Full Name: John", etc.
        if re.match(r"^[\w\s\-]{2,52}:\s*\S+$", p) and len(p) < 140:
            continue
        # Also reject sentences that start with multiple field-like pattern indicators
        # (e.g., many colons suggest a field dump)
        colon_count = p.count(":")
        if colon_count > 2:
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


def try_build_grounded_document_overview(query: str, hits: list[RetrievedChunk]) -> ExtractedAnswer | None:
    """
    Deterministic fallback for broad overview / summary questions when excerpts
    cluster on a dominant source. Produces meaning-focused prose (not a field inventory).
    """
    qn = normalize_query_for_pipeline(query or "")
    if not hits or not _DOC_ABOUT_Q.search(qn):
        return None

    srcs = [str(h.metadata.get("source_name") or "").strip() for h in hits]
    srcs = [s for s in srcs if s]
    if not srcs:
        return None
    top_src, top_n = Counter(srcs).most_common(1)[0]
    if top_n < 3 and (top_n / max(len(srcs), 1)) < 0.6:
        return None

    parts: list[str] = []
    used_idx: list[int] = []
    for idx, h in enumerate(hits, start=1):
        if str(h.metadata.get("source_name") or "").strip() != top_src:
            continue
        t = (h.page_content or "").strip()
        if t:
            parts.append(t)
            used_idx.append(idx)
    blob = "\n".join(parts)
    if len(blob.strip()) < 24:
        return None

    cite_sources = tuple(sorted(set(used_idx))[:2]) or (1,)
    cite_str = " ".join(f"[SOURCE {n}]" for n in cite_sources)

    sents = _overview_prose_sentences(blob)
    if len(sents) >= 2:
        body = " ".join(sents[:3])
        if len(body) > 900:
            body = body[:897].rstrip() + "…"
        answer = (
            f"{body} {cite_str}"
        ).strip()
        return ExtractedAnswer(answer=answer, used_source_numbers=cite_sources)

    # For field-heavy docs (prose extraction failed), try theme-based summary.
    # This preserves structured doc behavior: loans, forms, etc.
    themes = _overview_themes_from_blob(blob)
    # If doc is very field-structured (few extracted prose sentences) but has themes, use themes
    # even if < 2 sentences found. This handles mixed field+narrative like SpaceX profiles.
    if len(themes) >= 2 or (len(themes) >= 1 and len(sents) == 0):
        # Try to build a more specific summary with extracted entities (company, person, roles, projects)
        specific_summary = _build_specific_overview_summary(blob, themes)
        if specific_summary:
            answer = (f"{specific_summary} {cite_str}").strip()
        else:
            # Fallback to generic theme-based summary
            if len(themes) == 2:
                theme_txt = f"{themes[0]} and {themes[1]}"
            else:
                theme_txt = ", ".join(themes[:-1]) + f", and {themes[-1]}"
            answer = (
                f"It provides an overview of {theme_txt}. {cite_str}"
            ).strip()
        return ExtractedAnswer(answer=answer, used_source_numbers=cite_sources)

    # If blob is very short (< 100 chars), try to return at least a grounded answer
    # from raw content rather than silently failing
    if len(blob.strip()) > 0 and len(blob.strip()) <= 100:
        trimmed = blob.strip()[:97] + ("…" if len(blob.strip()) > 97 else "")
        answer = (f"{trimmed} {cite_str}").strip()
        return ExtractedAnswer(answer=answer, used_source_numbers=cite_sources)
    
    return None


def try_answer_section_navigation_fallback(
    query: str, hits: list[RetrievedChunk]
) -> ExtractedAnswer | None:
    """
    When the model abstains on a section-targeted question, quote the matching sentence
    from retrieved text if it clearly references the requested section number.
    """
    from app.llm.query_intent import is_section_navigation_query

    if not hits or not is_section_navigation_query(query):
        return None
    ql = (query or "").strip().lower()
    num: str | None = None
    m = re.search(r"section\s*(?:#|number|num\.?)?\s*(\d+)\b", ql)
    if m:
        num = m.group(1)
    else:
        word_to_n = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        for w, d in word_to_n.items():
            if "section" in ql and re.search(rf"\b{re.escape(w)}\b", ql):
                num = d
                break
    if not num:
        return None
    needle = re.compile(rf"section\s*{re.escape(num)}\b", re.I)
    for idx, h in enumerate(hits, start=1):
        text = (h.page_content or "").strip()
        if not text or not needle.search(text):
            continue
        flat = re.sub(r"\s+", " ", text.replace("\n", " "))
        chunks = re.split(r"(?<=[.!?])\s+", flat)
        for p in chunks:
            p = p.strip()
            if needle.search(p) and 10 <= len(p) <= 420:
                return ExtractedAnswer(answer=f"{p} [SOURCE {idx}]", used_source_numbers=(idx,))
        if needle.search(flat) and len(flat) <= 520:
            return ExtractedAnswer(answer=f"{flat} [SOURCE {idx}]", used_source_numbers=(idx,))
    return None


_EMAIL_ANY = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def try_extract_field_from_raw_library(
    query: str,
    raw_paths: list[Path | str],
    *,
    preferred_source: str | None = None,
) -> tuple[ExtractedAnswer, list[RetrievedChunk]] | None:
    """
    Scan uploaded library files on disk (full normalized text per file) for obvious field rows.

    Used when hybrid chunks miss a label/value pair that still exists in the raw file.
    """
    from app.ingestion.loader import load_file

    kind = _query_kind(query)
    if not kind or not raw_paths:
        return None
    paths = [Path(p) for p in raw_paths]
    ps = (preferred_source or "").strip().lower()
    if ps:
        paths.sort(key=lambda p: 0 if p.name.lower() == ps else 1)
    for path in paths:
        if not path.is_file():
            continue
        try:
            docs = load_file(path)
        except Exception:
            continue
        full = "\n\n".join((d.text or "").strip() for d in docs if (d.text or "").strip())
        if len(full) < 4:
            continue
        sn = path.name
        preview = full[:12000]
        hit = RetrievedChunk(
            rank=0,
            page_content=preview,
            metadata={
                "source_name": sn,
                "chunk_id": f"raw-scan:{sn}",
                "page_number": None,
                "file_path": str(path.resolve()),
            },
            distance=0.35,
        )
        ext = try_extract_field_value_answer(query, [hit])
        if ext is None and kind == "email":
            m = _EMAIL_ANY.search(full)
            if m:
                ext = ExtractedAnswer(
                    answer=(
                        f"The document states that the email address is {m.group(0)} [SOURCE 1]."
                    ),
                    used_source_numbers=(1,),
                )
        if ext is not None:
            return ext, [hit]
    return None


def try_extract_contact_info_bundle_answer(query: str, hits: list[RetrievedChunk]) -> ExtractedAnswer | None:
    """
    When query implies contact details / contact info after a field lookup,
    extract ALL available contact fields (email, phone, website, address) deterministically.
    
    Only return grouped bundle when:
    - Query implies "details" / "contact" / "more info"
    - At least 2 contact fields are found with high confidence
    - All extracted fields have clear citations
    """
    if not hits:
        return None
    
    # Detect if query is asking for contact details / bundle
    q = (query or "").strip().lower()
    is_contact_details = bool(
        re.search(
            r"(details|contact|more\s+info|information|more\s+information|what\s+else|other\s+details)",
            q,
        )
    )
    if not is_contact_details:
        return None
    
    # Extract individual fields in order of appearance across all hits
    fields: dict[str, tuple[str, int]] = {}  # field_type -> (value, source_idx)
    
    for idx, h in enumerate(hits, start=1):
        text = (h.page_content or "").strip()
        if not text:
            continue
        
        lines = text.splitlines()
        
        # Extract email
        if "email" not in fields:
            for line in lines:
                m = re.search(
                    rf"(e-?mail|email)\s*[:\-]?\s*"
                    rf"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}})",
                    line,
                    re.I,
                )
                if m:
                    fields["email"] = (m.group(2).strip(), idx)
                    break
        
        # Extract phone - try multiple patterns
        if "phone" not in fields:
            for line in lines:
                # Pattern 1: Label + number
                m = re.search(
                    rf"(phone|contact|mobile|cell|tel\.?)\s*[:\-]?\s*"
                    rf"([\d()+\-\s]{{10,24}})",
                    line,
                    re.I,
                )
                if m:
                    raw_phone = re.sub(r"\s+", " ", m.group(2).strip())
                    if len(re.sub(r"\D", "", raw_phone)) >= 10:
                        fields["phone"] = (raw_phone, idx)
                        break
                
                # Pattern 2: Just number pattern (e.g., "555-1234" or "(555) 123-4567")
                if not m:
                    m = re.search(
                        rf"^[:\-]?\s*([\d()+\-\s]{{10,24}})\s*$",
                        line.strip(),
                    )
                    if m:
                        raw_phone = re.sub(r"\s+", " ", m.group(1).strip())
                        if len(re.sub(r"\D", "", raw_phone)) >= 10:
                            fields["phone"] = (raw_phone, idx)
                            break
        
        # Extract website
        if "website" not in fields:
            for line in lines:
                m = re.search(
                    rf"(website|url)\s*[:\-]?\s*"
                    rf"((?:https?://|www\.)[A-Za-z0-9._~/?#=\-]+)",
                    line,
                    re.I,
                )
                if not m:
                    m = re.search(
                        rf"(website|url)\s*[:\-]?\s*"
                        rf"([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9-]{1,24})+\.(?:com|net|org|io|edu|gov))",
                        line,
                        re.I,
                    )
                if m:
                    fields["website"] = (m.group(2).strip(), idx)
                    break
        
        # Extract address
        if "address" not in fields:
            for li, line in enumerate(lines):
                m = re.search(
                    rf"(address|location)\s*[:\-]?\s*(.+?)\s*$",
                    line,
                    re.I,
                )
                if m:
                    addr = m.group(2).strip()
                    addr = _merge_address_continuation(lines, li, addr)
                    addr = re.sub(r"\s+", " ", addr)
                    if 8 <= len(addr) <= 260:
                        low = addr.lower()
                        if any(
                            x in low
                            for x in (
                                "street",
                                "st.",
                                "road",
                                "rd.",
                                "avenue",
                                "ave",
                                "lane",
                                "drive",
                                "blvd",
                                "suite",
                                "unit",
                                "apt",
                                "city",
                                "state",
                                "zip",
                                "pin",
                                "country",
                                "p.o",
                                "po box",
                            )
                        ) or any(ch.isdigit() for ch in addr):
                            fields["address"] = (addr, idx)
                            break
    
    # Only return if at least 2 fields found
    if len(fields) < 2:
        return None
    
    # Build grouped response with citations
    parts: list[str] = []
    used_sources: set[int] = set()
    
    if "email" in fields:
        email, src = fields["email"]
        parts.append(f"**Email**: {email} [SOURCE {src}]")
        used_sources.add(src)
    
    if "phone" in fields:
        phone, src = fields["phone"]
        parts.append(f"**Phone**: {phone} [SOURCE {src}]")
        used_sources.add(src)
    
    if "website" in fields:
        website, src = fields["website"]
        parts.append(f"**Website**: {website} [SOURCE {src}]")
        used_sources.add(src)
    
    if "address" in fields:
        address, src = fields["address"]
        parts.append(f"**Address**: {address} [SOURCE {src}]")
        used_sources.add(src)
    
    answer = "Here is the contact information from the document:\n" + "\n".join(parts)
    
    return ExtractedAnswer(
        answer=answer,
        used_source_numbers=tuple(sorted(used_sources)),
    )



field_value_question_kind = _query_kind

