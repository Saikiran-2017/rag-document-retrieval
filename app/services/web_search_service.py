"""
Web snippets for grounded answers. Uses DuckDuckGo HTML API (no API key).

Fetches a wider candidate set, ranks/filters for quality, cleans snippets, and exposes
a small high-trust set to the LLM (Phase 14).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# How many raw hits to pull before ranking (noise rejection happens here).
WEB_FETCH_CAP = 10
# Snippets actually passed to the model (fewer = sharper answers).
_DEFAULT_MAX_FOR_LLM = 2
_SNIPPET_LLM_CHARS = 280

_TIME_SENSITIVE_HINT = re.compile(
    r"\b(today|latest|news|current|recent|price|stock|market|weather|breaking|"
    r"who is|who was|when did|20[2-3]\d)\b",
    re.I,
)
_LEADING_FILLER = re.compile(
    r"^(please|can you|could you|tell me|i want to know|i need to know|help me)\s+",
    re.I,
)
_BOILERPLATE = re.compile(
    r"\b(read more|click here|sign up|subscribe|cookie policy|privacy policy)\b[^.]*\.?",
    re.I,
)
_MULTI_SPACE = re.compile(r"\s+")
_REPEAT_SENTENCE = re.compile(r"(\b[\w\s,]{12,60}\b)(\.\s*\1)+", re.I)


@dataclass(frozen=True)
class WebSnippet:
    title: str
    url: str
    snippet: str


def web_search_enabled() -> bool:
    return os.environ.get("WEB_SEARCH_ENABLED", "1").strip().lower() not in ("0", "false", "no")


def _max_for_llm() -> int:
    raw = os.environ.get("WEB_MAX_LLM_RESULTS", "").strip()
    if raw.isdigit():
        return max(1, min(5, int(raw)))
    return _DEFAULT_MAX_FOR_LLM


def shape_query_for_web(user_query: str, retrieval_query: str) -> str:
    """
    Short, search-friendly query: avoid echoing long chatty questions when a tight
    retrieval rewrite exists; add a year hint for time-sensitive questions.
    """
    u = (user_query or "").strip()
    r = (retrieval_query or u).strip()
    base = _LEADING_FILLER.sub("", u) or r
    if len(base) > 160 and len(r) <= 160:
        base = r
    elif len(base) > 200:
        base = base[:200].rsplit(" ", 1)[0]
    base = _MULTI_SPACE.sub(" ", base).strip()
    if _TIME_SENSITIVE_HINT.search(u) and not re.search(r"\b20[2-3]\d\b", base):
        base = f"{base} {date.today().year}".strip()
    return base[:220] if base else r[:220]


def normalize_web_identity(url: str) -> str:
    """Host + path for dedupe and allow-list checks (no scheme)."""
    u = (url or "").strip().rstrip(").,]")
    if u.startswith("//"):
        u = "https:" + u
    if not u.startswith(("http://", "https://")):
        return ""
    try:
        p = urlparse(u)
    except ValueError:
        return ""
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (p.path or "").rstrip("/")
    return f"{host}{path}".lower()


def _query_terms(q: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]{3,}", (q or "").lower())
        if t not in frozenset({"the", "and", "for", "are", "was", "what", "when", "how", "why", "this", "that"})
    }


def clean_snippet_text(text: str, *, max_chars: int = _SNIPPET_LLM_CHARS) -> str:
    t = _MULTI_SPACE.sub(" ", (text or "").replace("\n", " ").replace("\r", " ")).strip()
    t = _BOILERPLATE.sub(" ", t)
    t = _REPEAT_SENTENCE.sub(r"\1", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    if len(t) > max_chars:
        cut = t[: max_chars + 1]
        if " " in cut:
            cut = cut[: max_chars].rsplit(" ", 1)[0]
        t = cut.rstrip(",;:") + ("…" if len(t) > max_chars else "")
    return t


def _score_candidate(s: WebSnippet, terms: set[str]) -> float:
    title_l = (s.title or "").lower()
    snip_l = (s.snippet or "").lower()
    blob = f"{title_l} {snip_l}"
    score = 0.0
    for t in terms:
        if t in blob:
            score += 1.1
    score += min(len(s.snippet), 320) / 320 * 2.4
    score += min(len(s.title), 72) / 72 * 1.2
    if len(s.snippet) < 32:
        score -= 2.5
    if len(s.title) < 6:
        score -= 0.8
    if re.fullmatch(r"[\W\d_]+", s.title or ""):
        score -= 3.0
    if "..." == (s.snippet or "").strip() or not (s.snippet or "").strip():
        score -= 4.0
    return score


def rank_and_filter_snippets(
    candidates: Iterable[WebSnippet],
    shaped_query: str,
    *,
    max_keep: int,
) -> list[WebSnippet]:
    terms = _query_terms(shaped_query)
    scored: list[tuple[float, WebSnippet]] = []
    seen_id: set[str] = set()
    for s in candidates:
        ident = normalize_web_identity(s.url)
        if not ident or ident in seen_id:
            continue
        seen_id.add(ident)
        sc = _score_candidate(s, terms)
        if sc < 0.35 and len(s.snippet) < 40:
            continue
        cleaned = WebSnippet(
            title=(s.title or s.url).strip() or s.url,
            url=s.url.strip(),
            snippet=clean_snippet_text(s.snippet),
        )
        if not cleaned.snippet and len(cleaned.title) < 10:
            continue
        scored.append((sc, cleaned))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [s for _, s in scored[:max_keep]]
    return out


def web_results_strong_enough(snippets: list[WebSnippet], *, shaped_query: str) -> bool:
    """True when at least one result looks worth citing (avoids hollow web-grounded replies)."""
    if not snippets:
        return False
    terms = _query_terms(shaped_query)
    for s in snippets:
        sc = _score_candidate(s, terms)
        if len(s.snippet) >= 52:
            sc += 0.35
        if sc >= 1.2:
            return True
    return False


def _ddgs_fetch(query: str, *, max_results: int) -> list[WebSnippet]:
    from duckduckgo_search import DDGS

    out: list[WebSnippet] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query.strip(), max_results=max_results):
            title = str(r.get("title") or "").strip()
            url = str(r.get("href") or r.get("url") or "").strip()
            body = str(r.get("body") or "").strip()
            if url and url.startswith("http"):
                out.append(WebSnippet(title=title or url, url=url, snippet=body))
    return out


def gather_ranked_web_snippets(
    user_query: str,
    retrieval_query: str,
) -> tuple[list[WebSnippet], str]:
    """
    Shape query, fetch candidates, rank. Returns (snippets, shaped_query) for strength checks.
    """
    if not web_search_enabled():
        return [], ""
    shaped = shape_query_for_web(user_query, retrieval_query)
    if not shaped.strip():
        return [], shaped
    try:
        raw = _ddgs_fetch(shaped, max_results=WEB_FETCH_CAP)
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return [], shaped
    snippets = rank_and_filter_snippets(raw, shaped, max_keep=_max_for_llm())
    return snippets, shaped


def search_web_for_chat(user_query: str, retrieval_query: str) -> list[WebSnippet]:
    """Shape query, fetch WEB_FETCH_CAP candidates, rank, return top WEB_MAX_LLM_RESULTS."""
    snippets, _ = gather_ranked_web_snippets(user_query, retrieval_query)
    return snippets


def search_web(query: str, *, max_results: int = 3) -> list[WebSnippet]:
    """Backward-compatible: ranked pipeline; ``max_results`` caps the returned list."""
    snippets, _ = gather_ranked_web_snippets(query, query)
    cap = max(1, min(5, max_results))
    return snippets[:cap]


def format_web_context(snippets: list[WebSnippet]) -> str:
    blocks: list[str] = []
    for i, s in enumerate(snippets, start=1):
        blocks.append(
            f"[WEB {i}]\nTitle: {s.title}\nURL: {s.url}\nSnippet:\n{s.snippet}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "(No web results.)"


def web_snippets_to_ui_payload(snippets: list[WebSnippet]) -> list[dict[str, str]]:
    """UI + persistence: same order and labels as [WEB n] in the prompt."""
    out: list[dict[str, str]] = []
    for i, s in enumerate(snippets, start=1):
        out.append(
            {
                "web_ref": f"WEB {i}",
                "title": s.title,
                "url": s.url,
                "snippet": s.snippet[: min(500, len(s.snippet))],
            }
        )
    return out


def prepare_web_for_generation(
    user_query: str,
    retrieval_query: str,
) -> tuple[list[WebSnippet], str, list[dict[str, str]], str]:
    """
    End-to-end: ranked snippets, LLM block, UI dicts (aligned indices), shaped query.
    """
    snippets, shaped = gather_ranked_web_snippets(user_query, retrieval_query)
    block = format_web_context(snippets)
    payload = web_snippets_to_ui_payload(snippets)
    return snippets, block, payload, shaped
