"""Rewrite user queries for better retrieval (single fast LLM call, optional)."""

from __future__ import annotations

import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_openai_api_key

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = """You rewrite a user question into a short search query for a document or knowledge base.
Rules:
- Output ONE line only: the rewritten query. No quotes, no explanation.
- Keep key entities, names, dates, and technical terms.
- If the message is already a good search query, return it nearly unchanged."""


def should_skip_rewrite_llm(query: str) -> bool:
    """
    Heuristic skip to avoid an extra LLM round-trip for short or keyword-like queries.

    Still respects ``KA_NO_REWRITE=1`` inside :func:`rewrite_for_retrieval`.
    """
    q = query.strip()
    if len(q) < 8:
        return True
    if len(q) <= 52 and len(q.split()) <= 6:
        return True
    if len(q) <= 80 and "?" not in q and len(q.split()) <= 12:
        return True
    if len(q) <= 36 and not re.search(r"\b(why|how|explain|compare|summarize|difference)\b", q, re.I):
        return True
    return False


def rewrite_for_retrieval(user_query: str, *, model: str = "gpt-4o-mini") -> str:
    """
    Return a retrieval-oriented query string.

    Skips the LLM when ``KA_NO_REWRITE=1``, query is very short, or :func:`should_skip_rewrite_llm` is true.
    """
    if os.environ.get("KA_NO_REWRITE", "").strip().lower() in ("1", "true", "yes"):
        return user_query.strip()
    q = user_query.strip()
    if should_skip_rewrite_llm(q):
        return q
    try:
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=80,
            api_key=get_openai_api_key(),
        )
        out = llm.invoke(
            [
                SystemMessage(content=REWRITE_SYSTEM),
                HumanMessage(content=f"User question:\n{q}"),
            ]
        )
        text = (out.content or "").strip() if hasattr(out, "content") else str(out).strip()
        if isinstance(out.content, list):
            text = "".join(
                str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in out.content
            ).strip()
        line = text.split("\n")[0].strip().strip('"').strip("'")
        return line if len(line) >= 3 else q
    except Exception as exc:
        logger.warning("Query rewrite failed, using original: %s", exc)
        return q
