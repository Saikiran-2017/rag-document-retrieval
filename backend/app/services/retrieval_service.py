"""Re-exports hybrid retrieval and context shaping (BM25 + vector + rerank)."""

from app.retrieval.context_selection import (
    hybrid_pool_size,
    rerank_hybrid_hits,
    select_generation_context,
)
from app.retrieval.hybrid_retrieve import hybrid_retrieve

__all__ = [
    "hybrid_pool_size",
    "hybrid_retrieve",
    "rerank_hybrid_hits",
    "select_generation_context",
]
