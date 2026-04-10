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

_FIELD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "loan_disbursed_amount",
        re.compile(
            rf"\b(loan{_WS}disbursed{_WS}amount|disbursed{_WS}amount|loan{_WS}amount)\b{_WS}[:\-]?\s*({_AMOUNT})\b",
            re.I,
        ),
    ),
    (
        "person_name",
        re.compile(
            rf"\b("
            rf"applicant{_WS}name|full{_WS}name|subject{_WS}name|borrower{_WS}name|primary{_WS}holder|"
            rf"customer{_WS}name|account{_WS}holder|legal{_WS}name"
            rf")\b{_WS}[:\-]?\s*([A-Z][A-Za-z \.\'\-]{{2,}})\b",
            re.I,
        ),
    ),
    (
        "application_number",
        re.compile(
            rf"\b(application{_WS}(number|no\.?|#))\b{_WS}[:\-]?\s*({_ID})\b",
            re.I,
        ),
    ),
    (
        "email",
        re.compile(
            rf"\b(e-?mail|email)\b{_WS}[:\-]?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}})\b",
            re.I,
        ),
    ),
    (
        "phone",
        re.compile(
            rf"\b(contact|phone|mobile|cell|tel\.?)\s*(number)?\b{_WS}[:\-]?\s*([\d()+\-\s]{{10,24}})\b",
            re.I,
        ),
    ),
    (
        "address",
        re.compile(
            rf"\b(current\s+)?address\b{_WS}[:\-]?\s*(.+?)\s*$",
            re.I,
        ),
    ),
]


def _query_kind(query: str) -> str | None:
    q = (query or "").strip().lower()
    if not q:
        return None
    if re.search(r"\b(his|her|their)\s+name\b|\bwhat\s+is\s+(his|her|their)\s+name\b", q):
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
    if re.search(r"\b(e-?mail|email address)\b", q):
        return "email"
    if re.search(r"\b(phone|mobile|contact number|cell)\b", q):
        return "phone"
    if re.search(r"\b(current\s+)?address\b|\bstreet address\b|\bmailing address\b", q):
        return "address"
    return None


def try_extract_field_value_answer(query: str, hits: list[RetrievedChunk]) -> ExtractedAnswer | None:
    """
    If query looks like a field/value lookup (loan amount, applicant name, etc.) and a clear
    label/value pair exists in any selected chunk, return a grounded answer with a citation.
    """
    kind = _query_kind(query)
    if not kind or not hits:
        return None
    pat = dict(_FIELD_PATTERNS).get(kind)
    if pat is None:
        return None

    for idx, h in enumerate(hits, start=1):
        text = (h.page_content or "").strip()
        if not text:
            continue
        # Scan line-by-line to avoid accidental cross-line joins.
        for line in text.splitlines():
            m = pat.search(line)
            if not m:
                continue
            if kind == "loan_disbursed_amount":
                amount = m.group(2)
                return ExtractedAnswer(
                    answer=f"Loan disbursed amount: {amount} [SOURCE {idx}]",
                    used_source_numbers=(idx,),
                )
            if kind == "person_name":
                name = m.group(2).strip()
                # Avoid overly generic false positives.
                parts = [p for p in name.split() if p]
                if len(name) < 3 or len(parts) < 1:
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
                    answer=f"Name on record: {name} [SOURCE {idx}]",
                    used_source_numbers=(idx,),
                )
            if kind == "application_number":
                app_no = m.group(3).strip()
                return ExtractedAnswer(
                    answer=f"Application number: {app_no} [SOURCE {idx}]",
                    used_source_numbers=(idx,),
                )
            if kind == "email":
                em = m.group(2).strip()
                return ExtractedAnswer(
                    answer=f"Email: {em} [SOURCE {idx}]",
                    used_source_numbers=(idx,),
                )
            if kind == "phone":
                raw_phone = re.sub(r"\s+", " ", m.group(3).strip())
                if len(re.sub(r"\D", "", raw_phone)) < 10:
                    continue
                return ExtractedAnswer(
                    answer=f"Contact number: {raw_phone} [SOURCE {idx}]",
                    used_source_numbers=(idx,),
                )
            if kind == "address":
                addr = m.group(2).strip()
                addr = re.sub(r"\s+", " ", addr)
                if len(addr) < 8 or len(addr) > 220:
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
                    )
                ) and not any(ch.isdigit() for ch in addr):
                    continue
                return ExtractedAnswer(
                    answer=f"Address on record: {addr} [SOURCE {idx}]",
                    used_source_numbers=(idx,),
                )
    return None


_METADATA_FILENAME_Q = re.compile(
    r"\b("
    r"file\s*name|filename|document\s*name|name\s+of\s+(the\s+)?(file|document)|"
    r"which\s+file|what\s+file|what\s+is\s+the\s+(file|document)|"
    r"which\s+document|source\s+file"
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
    r"summarize\s+this\s+(file|document)|what\s+does\s+this\s+(file|document)\s+contain|"
    r"what\s+is\s+in\s+this\s+(file|document))\b",
    re.I,
)

_FIELD_LABEL = re.compile(
    r"^\s*([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,6})\s*[:\-]\s*.+$"
)


def try_build_grounded_document_overview(query: str, hits: list[RetrievedChunk]) -> ExtractedAnswer | None:
    """
    Deterministic fallback for broad "what is this document about?" questions when:
    - retrieval selected a dominant document, and
    - the excerpts expose obvious structure (field labels / section-ish lines).
    """
    if not hits or not _DOC_ABOUT_Q.search(query or ""):
        return None

    srcs = [str(h.metadata.get("source_name") or "").strip() for h in hits]
    srcs = [s for s in srcs if s]
    if not srcs:
        return None
    top_src, top_n = Counter(srcs).most_common(1)[0]
    if top_n < 3 and (top_n / max(len(srcs), 1)) < 0.6:
        return None

    labels: list[str] = []
    used_sources: list[int] = []
    for idx, h in enumerate(hits, start=1):
        if str(h.metadata.get("source_name") or "").strip() != top_src:
            continue
        for line in (h.page_content or "").splitlines():
            m = _FIELD_LABEL.match(line)
            if not m:
                continue
            lab = m.group(1).strip()
            # Filter out very short / noisy labels.
            if len(lab) < 6:
                continue
            labels.append(lab)
            used_sources.append(idx)
            if len(labels) >= 10:
                break
        if len(labels) >= 10:
            break

    # Also accept "label-only" lines that look like UI field names (no colon),
    # commonly seen in intranet screenshots extracted to text.
    if len(labels) < 4:
        for idx, h in enumerate(hits, start=1):
            if str(h.metadata.get("source_name") or "").strip() != top_src:
                continue
            for line in (h.page_content or "").splitlines():
                t = line.strip()
                if not t or len(t) < 8 or len(t) > 64:
                    continue
                if re.match(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,6}$", t):
                    labels.append(t)
                    used_sources.append(idx)
                    if len(labels) >= 10:
                        break
            if len(labels) >= 10:
                break

    uniq: list[str] = []
    seen: set[str] = set()
    for l in labels:
        key = l.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(l)
        if len(uniq) >= 8:
            break
    if len(uniq) < 3:
        return None

    # Cite up to 2 sources that contain label evidence (keep it minimal).
    cite_sources = tuple(sorted(set(used_sources))[:2]) or (1,)
    cite_str = " ".join(f"[SOURCE {n}]" for n in cite_sources)
    bullets = "\n".join(f"- {l}" for l in uniq[:8])
    answer = (
        f"This document appears to be a structured administrative record with labeled fields and sections {cite_str}.\n\n"
        f"Key items it contains (from the retrieved excerpts):\n{bullets}\n\n"
        f"{cite_str}"
    ).strip()
    return ExtractedAnswer(answer=answer, used_source_numbers=cite_sources)


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

