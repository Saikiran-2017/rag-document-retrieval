"""
Hybrid retrieval: dense (FAISS) + sparse (BM25) fused with RRF, then top-k for generation.

Improves recall vs vector-only; final ordering is explainable in debug via rrf_score metadata.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from app.retrieval.vector_store import (
    RetrievedChunk,
    faiss_vector_count,
    iter_faiss_documents,
    retrieve_top_k,
)

# Reciprocal Rank Fusion depth (standard RRF k).
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _chunk_key(meta: dict[str, Any], fallback: str) -> str:
    cid = meta.get("chunk_id")
    if cid not in (None, ""):
        return str(cid)
    return fallback


def hybrid_retrieve(
    store: FAISS,
    query: str,
    *,
    k_final: int,
    k_vector: int = 24,
    k_bm25: int = 24,
) -> list[RetrievedChunk]:
    """
    Retrieve ``k_final`` chunks after BM25 + vector search and RRF merge.

    Each hit's ``metadata`` gains ``rrf_score``, ``vector_rank`` (optional), ``bm25_rank`` (optional).
    ``distance`` on the result is the **vector** distance when the chunk appeared in vector results;
    otherwise a sentinel high distance for BM25-only hits (caller should use ``rrf_score`` + hybrid gate).
    """
    if not query.strip():
        raise ValueError("Query must be non-empty.")

    nvec = faiss_vector_count(store)
    if nvec == 0:
        return []

    docs = iter_faiss_documents(store)
    if not docs:
        kv = min(max(k_final, 4), nvec)
        return retrieve_top_k(store, query, k=kv)

    corpus_tokens: list[list[str]] = []
    keys: list[str] = []
    for i, doc in enumerate(docs):
        meta = dict(doc.metadata or {})
        key = _chunk_key(meta, f"row_{i}")
        keys.append(key)
        corpus_tokens.append(_tokenize(doc.page_content))

    bm25 = BM25Okapi(corpus_tokens)
    q_tok = _tokenize(query)
    bm25_scores = bm25.get_scores(q_tok) if q_tok else [0.0] * len(keys)
    bm25_order = sorted(range(len(keys)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank: dict[str, int] = {}
    for r, idx in enumerate(bm25_order[: max(k_bm25, k_final * 3)]):
        bm25_rank[keys[idx]] = r + 1

    kv = min(k_vector, nvec)
    vec_hits = retrieve_top_k(store, query, k=kv)
    vec_rank: dict[str, int] = {}
    vec_dist: dict[str, float] = {}
    vec_by_key: dict[str, RetrievedChunk] = {}
    for r, h in enumerate(vec_hits):
        k = _chunk_key(h.metadata, "")
        if k:
            vec_rank[k] = r + 1
            vec_dist[k] = float(h.distance)
            vec_by_key[k] = h

    key_to_doc: dict[str, tuple[str, dict[str, Any]]] = {}
    for i, doc in enumerate(docs):
        meta = dict(doc.metadata or {})
        key = keys[i]
        key_to_doc[key] = (doc.page_content, meta)

    all_keys = set(vec_rank) | set(bm25_rank)
    rrf: dict[str, float] = {key: 0.0 for key in all_keys}
    for key in all_keys:
        if key in vec_rank:
            rrf[key] += 1.0 / (_RRF_K + vec_rank[key])
        if key in bm25_rank:
            rrf[key] += 1.0 / (_RRF_K + bm25_rank[key])

    ranked_keys = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)
    kf = min(k_final, len(ranked_keys))
    out: list[RetrievedChunk] = []
    for rank, key in enumerate(ranked_keys[:kf]):
        pair = key_to_doc.get(key)
        if pair:
            content, meta = pair
            meta = dict(meta)
        elif key in vec_by_key:
            h = vec_by_key[key]
            content, meta = h.page_content, dict(h.metadata)
        else:
            content, meta = "", {}
        meta["rrf_score"] = round(rrf[key], 6)
        if key in vec_rank:
            meta["vector_rank"] = vec_rank[key]
        if key in bm25_rank:
            meta["bm25_rank"] = bm25_rank[key]
        dist = float(vec_dist.get(key, 2.5))
        out.append(
            RetrievedChunk(
                rank=rank,
                page_content=content,
                metadata=meta,
                distance=dist,
            )
        )
    return out
