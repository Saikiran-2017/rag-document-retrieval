"""ensure_index_matches_library must succeed outside Streamlit when rebuild returns ok."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services import index_service


def test_ensure_returns_true_when_rebuild_ok_without_streamlit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    faiss = tmp_path / "faiss"
    raw.mkdir()
    faiss.mkdir()
    (raw / "doc.txt").write_text("sample corpus text for fingerprint", encoding="utf-8")

    monkeypatch.setattr(index_service, "_streamlit_script_running", lambda: False)
    monkeypatch.setattr(
        index_service,
        "rebuild_knowledge_index",
        lambda *a, **k: (True, "", 3, "rebuilt"),
    )

    ok, msg = index_service.ensure_index_matches_library(
        raw, faiss, chunk_size=500, chunk_overlap=50
    )
    assert ok is True
    assert msg == ""
