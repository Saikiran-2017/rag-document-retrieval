"""
Lightweight on-disk hints about likely extraction quality (no full ingestion).

Used for catalog / UI only; real chunking still goes through :mod:`app.ingestion.loader`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

ExtractionQuality = Literal["good", "low_text", "unknown"]


def lightweight_extraction_signal(path: Path) -> dict[str, Any]:
    """
    Return ``quality`` (``good`` | ``low_text`` | ``unknown``) and optional ``note`` for UI.

    PDFs use a quick pypdf sample (first pages only). DOCX opens the file briefly.
    """
    if not path.is_file():
        return {"quality": "unknown", "note": "File not found."}

    suf = path.suffix.lower()
    if suf == ".pdf":
        q, note = _pdf_quick_signal(path)
        return {"quality": q, "note": note}
    if suf == ".docx":
        q, note = _docx_quick_signal(path)
        return {"quality": q, "note": note}
    if suf == ".txt":
        q, note = _txt_quick_signal(path)
        return {"quality": q, "note": note}
    return {"quality": "unknown", "note": None}


def _pdf_quick_signal(path: Path) -> tuple[ExtractionQuality, str | None]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "unknown", None
    try:
        reader = PdfReader(str(path))
        n = len(reader.pages)
        if n == 0:
            return "low_text", "PDF has no pages."
        sample = min(3, n)
        total = 0
        for i in range(sample):
            try:
                total += len((reader.pages[i].extract_text() or "").strip())
            except Exception:
                pass
        avg = total / sample if sample else 0.0
        if avg < 35:
            return (
                "low_text",
                "Little selectable text — scanned or image-heavy PDFs may need optional OCR.",
            )
        return "good", None
    except Exception as exc:
        logger.debug("pdf quick signal %s: %s", path.name, exc)
        return "unknown", "Could not preview PDF text."


def _docx_quick_signal(path: Path) -> tuple[ExtractionQuality, str | None]:
    try:
        from docx import Document as DocxDocument

        d = DocxDocument(path)
        tlen = sum(len(p.text) for p in d.paragraphs)
        if tlen < 40:
            return "low_text", "Very little text in this DOCX."
        return "good", None
    except Exception as exc:
        logger.debug("docx quick signal %s: %s", path.name, exc)
        return "unknown", "Could not preview DOCX."


def _txt_quick_signal(path: Path) -> tuple[ExtractionQuality, str | None]:
    try:
        with path.open("rb") as f:
            chunk = f.read(256 * 1024)
        text = chunk.decode("utf-8", errors="replace")
        if len(text.strip()) < 15:
            return "low_text", "Very little text in this file."
        return "good", None
    except OSError:
        return "unknown", None
