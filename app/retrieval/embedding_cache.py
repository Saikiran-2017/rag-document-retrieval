"""Disk cache for embedding vectors keyed by SHA-256 of UTF-8 text."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
import numpy as np
from langchain_openai import OpenAIEmbeddings


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_model_dirname(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)[:120]


def cache_dir_for(faiss_folder: Path, embedding_model: str) -> Path:
    p = Path(faiss_folder).resolve() / "embedding_cache" / _safe_model_dirname(embedding_model)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path_for_hash(cache_root: Path, h: str) -> Path:
    return cache_root / f"{h}.npy"


def load_cached_vector(cache_root: Path, h: str) -> list[float] | None:
    path = _path_for_hash(cache_root, h)
    if not path.is_file():
        return None
    try:
        arr = np.load(path)
        return arr.astype(np.float64).tolist()
    except OSError:
        return None


def save_vector(cache_root: Path, h: str, vector: list[float]) -> None:
    path = _path_for_hash(cache_root, h)
    np.save(path, np.array(vector, dtype=np.float32), allow_pickle=False)


def embed_texts_with_cache(
    texts: list[str],
    embedder: OpenAIEmbeddings,
    cache_root: Path,
    *,
    batch_size: int = 64,
) -> list[list[float]]:
    """
    Return embedding vectors in the same order as ``texts``.

    Uses ``embedder.embed_documents`` only for texts whose content hash is not on disk.
    """
    if not texts:
        return []
    cache_root.mkdir(parents=True, exist_ok=True)
    hashes = [content_hash(t) for t in texts]
    out: list[list[float] | None] = [None] * len(texts)
    missing_indices: list[int] = []
    for i, h in enumerate(hashes):
        cached = load_cached_vector(cache_root, h)
        if cached is not None:
            out[i] = cached
        else:
            missing_indices.append(i)
    if missing_indices:
        to_embed = [texts[i] for i in missing_indices]
        for start in range(0, len(to_embed), batch_size):
            batch = to_embed[start : start + batch_size]
            batch_vecs = embedder.embed_documents(batch)
            for j, vec in enumerate(batch_vecs):
                global_i = missing_indices[start + j]
                h = hashes[global_i]
                out[global_i] = vec
                save_vector(cache_root, h, vec)
    return [v for v in out if v is not None]
