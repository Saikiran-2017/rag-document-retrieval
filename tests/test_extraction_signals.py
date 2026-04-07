"""Phase 28: lightweight extraction preview signals (catalog / UI)."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.extraction_signals import lightweight_extraction_signal
from app.services import index_service
from app.services.library_catalog import documents_list_api_payload


def test_txt_good_signal(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("Enough text here for a good signal about extraction.\n", encoding="utf-8")
    sig = lightweight_extraction_signal(p)
    assert sig["quality"] == "good"


def test_txt_low_text_signal(tmp_path: Path) -> None:
    p = tmp_path / "b.txt"
    p.write_text("x", encoding="utf-8")
    sig = lightweight_extraction_signal(p)
    assert sig["quality"] == "low_text"


def test_library_needs_user_sync_empty_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    faiss = tmp_path / "faiss"
    raw.mkdir()
    faiss.mkdir()
    assert index_service.library_needs_user_sync(raw, faiss) is False


def test_documents_list_payload_includes_flags(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    faiss = tmp_path / "faiss"
    raw.mkdir()
    faiss.mkdir()
    (raw / "note.txt").write_text("hello world " * 20, encoding="utf-8")
    payload = documents_list_api_payload(raw, faiss)
    assert payload["count"] == 1
    assert payload["library_needs_sync"] is True
    row = payload["documents"][0]
    assert row["extraction_quality"] == "good"
    assert "filename" in row
