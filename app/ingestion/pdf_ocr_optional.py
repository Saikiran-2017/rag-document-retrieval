"""
Opt-in OCR for PDF pages when normal text extraction returns nothing.

Disabled by default. Set ``RAG_ENABLE_PDF_OCR=1`` and install optional deps
(see ``requirements-optional-ocr.txt``). Requires a working Tesseract binary
on PATH and Poppler for ``pdf2image`` (platform-specific).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def pdf_ocr_enabled() -> bool:
    v = os.environ.get("RAG_ENABLE_PDF_OCR", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def try_ocr_pdf_page(path: Path, page_number_1based: int) -> str:
    """
    Rasterize a single PDF page and run Tesseract. Returns normalized single-line-ish text or "".
    """
    if not pdf_ocr_enabled():
        return ""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        logger.info(
            "RAG_ENABLE_PDF_OCR is set but optional OCR imports failed "
            "(install pdf2image, pytesseract, Pillow; see requirements-optional-ocr.txt)."
        )
        return ""
    try:
        images = convert_from_path(
            path,
            first_page=page_number_1based,
            last_page=page_number_1based,
            dpi=200,
        )
        if not images:
            return ""
        raw = pytesseract.image_to_string(images[0], lang="eng") or ""
        return " ".join(raw.split())
    except Exception as exc:
        logger.warning("OCR failed for %s page %s: %s", path.name, page_number_1based, exc)
        return ""
