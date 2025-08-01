"""
Split ingested documents into overlapping text chunks for embedding and retrieval.

Uses LangChain's RecursiveCharacterTextSplitter so splits prefer natural boundaries
(paragraphs, sentences) before hard character cuts—easy to justify in interviews.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loader import IngestedDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextChunk:
    """
    One text segment produced from a single :class:`IngestedDocument` (e.g. one PDF page).

    ``chunk_index`` and ``total_chunks`` apply **within that segment only** (not the whole file).

    ``metadata`` is a plain dict of JSON-friendly values (str, int, None) suitable for
    LangChain ``Document(metadata=...)`` and FAISS index serialization.
    """

    text: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any]


def _validate_split_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size (otherwise chunks repeat infinitely).")


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """RecursiveCharacterTextSplitter applies overlap between consecutive chunks internally."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_ingested_documents(
    documents: list[IngestedDocument],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """
    Split each ingested segment into overlapping chunks.

    For every non-empty ``IngestedDocument``, ``RecursiveCharacterTextSplitter`` splits
    ``doc.text`` with the requested character overlap, then each piece becomes one
    ``TextChunk`` with ``chunk_index`` in ``0 .. total_chunks - 1`` for **that** segment.

    Parameters
    ----------
    documents
        Output from :func:`app.ingestion.loader.load_file` or ``load_raw_directory``.
    chunk_size
        Target maximum characters per chunk (not tokens—embedding models have their own limits).
    chunk_overlap
        Characters repeated at chunk boundaries; implemented by LangChain, not duplicated here.

    Returns
    -------
    Flat list of ``TextChunk`` in stable order: all chunks for document 0, then document 1, etc.
    """
    _validate_split_params(chunk_size, chunk_overlap)
    splitter = _make_splitter(chunk_size, chunk_overlap)
    out: list[TextChunk] = []

    for doc in documents:
        if not doc.text or not doc.text.strip():
            logger.debug("Skipping empty document: %s", doc.document_id)
            continue

        raw_pieces = splitter.split_text(doc.text)
        pieces = [p.strip() for p in raw_pieces if p.strip()]
        if not pieces:
            continue

        total = len(pieces)
        base_meta = dict(doc.metadata_dict)

        for i, text in enumerate(pieces):
            chunk_id = f"{doc.document_id}_c{i:04d}"
            metadata = {
                **base_meta,
                "document_id": doc.document_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": total,
            }
            out.append(
                TextChunk(
                    text=text,
                    chunk_id=chunk_id,
                    chunk_index=i,
                    total_chunks=total,
                    metadata=metadata,
                )
            )

    return out


def print_chunk_summary(chunks: list[TextChunk], *, preview_first: int = 3) -> None:
    """Print counts and short previews for manual testing."""
    print(f"Total chunks: {len(chunks)}")
    if not chunks:
        return

    by_doc: dict[str, int] = {}
    for c in chunks:
        did = str(c.metadata.get("document_id", "?"))
        by_doc[did] = by_doc.get(did, 0) + 1

    print("Chunks per document_id:")
    for did, n in sorted(by_doc.items()):
        print(f"  {did}: {n}")

    n_show = min(preview_first, len(chunks))
    print(f"\nFirst {n_show} chunk preview(s):")
    for c in chunks[:n_show]:
        prev = c.text[:160] + ("…" if len(c.text) > 160 else "")
        print(f"\n  chunk_id={c.chunk_id}  chunk_index={c.chunk_index}  total_chunks={c.total_chunks}")
        print(f"  {prev!r}")


def chunk_single_file(
    path: str | Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Load one file via ingestion, then chunk (handy for quick tests)."""
    from app.ingestion.loader import load_file

    return chunk_ingested_documents(
        load_file(path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_raw_directory(
    raw_dir: str | Path | None = None,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Load all supported files from ``data/raw`` (or ``raw_dir``), then chunk."""
    from app.ingestion.loader import load_raw_directory

    return chunk_ingested_documents(
        load_raw_directory(raw_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from app.ingestion.loader import get_default_raw_dir, load_file, load_raw_directory

    default_raw = get_default_raw_dir()
    sample_txt = default_raw / "sample_phase2.txt"

    chunk_size, overlap = 80, 20  # small demo so short sample splits into multiple chunks

    print("=== 1) Chunk one file (sample_phase2.txt if present) ===\n")
    if sample_txt.is_file():
        one_file_chunks = chunk_ingested_documents(
            load_file(sample_txt),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        print_chunk_summary(one_file_chunks, preview_first=5)
    else:
        print(f"Missing {sample_txt}; skip single-file demo.\n")

    print("\n=== 2) Chunk everything under data/raw/ ===\n")
    all_docs = load_raw_directory(default_raw)
    all_chunks = chunk_ingested_documents(
        all_docs,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    print_chunk_summary(all_chunks, preview_first=5)

    if not all_chunks:
        print(
            "\n(No chunks produced. Add non-empty .pdf/.docx/.txt under data/raw/ or lower chunk_size in __main__.)"
        )

# Fixed 2024-08-08

# Fixed 2024-09-04

# Fixed 2024-09-26

# Fixed 2024-10-19

# Fixed 2024-08-08

# Fixed 2024-09-04
