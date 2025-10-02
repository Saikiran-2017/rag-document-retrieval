"""
Phase 15: routing contracts for chat_service.answer_user_query (mocked I/O, no API keys required).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services import chat_service


@pytest.fixture
def lib_dirs(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    faiss = tmp_path / "faiss"
    raw.mkdir()
    faiss.mkdir()
    return raw, faiss


def _capture_routing(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    out: list[str] = []
    real = chat_service._finalize_answer

    def _wrap(turn, **dbg):
        out.append(str(dbg.get("routing", "")))
        return real(turn, **dbg)

    monkeypatch.setattr(chat_service, "_finalize_answer", _wrap)
    return out


def test_empty_query_error(lib_dirs, monkeypatch):
    _capture_routing(monkeypatch)
    raw, faiss = lib_dirs
    t = chat_service.answer_user_query(
        "   ",
        raw_dir=raw,
        faiss_folder=faiss,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )
    assert t.mode == "error"


def test_general_no_library(lib_dirs, monkeypatch):
    routes = _capture_routing(monkeypatch)
    monkeypatch.setattr(chat_service, "safe_general_answer", lambda q: ("ok", None))
    raw, faiss = lib_dirs
    chat_service.answer_user_query(
        "hello there",
        raw_dir=raw,
        faiss_folder=faiss,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )
    assert routes == ["general_no_library"]


def test_general_sync_fallback(lib_dirs, monkeypatch):
    routes = _capture_routing(monkeypatch)
    monkeypatch.setattr(chat_service, "safe_general_answer", lambda q: ("ok", None))
    (lib_dirs[0] / "a.txt").write_text("x", encoding="utf-8")

    def _list(rd: Path):
        return sorted(rd.glob("*"))

    monkeypatch.setattr(chat_service.index_service, "list_raw_files", _list)
    monkeypatch.setattr(
        chat_service.index_service,
        "ensure_index_matches_library",
        lambda *a, **k: (False, "fail"),
    )
    raw, faiss = lib_dirs
    chat_service.answer_user_query(
        "question",
        raw_dir=raw,
        faiss_folder=faiss,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )
    assert routes == ["general_sync_fallback"]


def test_general_retrieval_failed_store(lib_dirs, monkeypatch):
    routes = _capture_routing(monkeypatch)
    monkeypatch.setattr(chat_service, "safe_general_answer", lambda q: ("ok", None))
    (lib_dirs[0] / "a.txt").write_text("content", encoding="utf-8")

    def _list(rd: Path):
        return sorted(rd.glob("*"))

    monkeypatch.setattr(chat_service.index_service, "list_raw_files", _list)
    monkeypatch.setattr(
        chat_service.index_service,
        "ensure_index_matches_library",
        lambda *a, **k: (True, ""),
    )
    monkeypatch.setattr(chat_service, "wants_no_retrieval_fastpath", lambda q: False)
    monkeypatch.setattr(
        chat_service.index_service,
        "load_faiss_store",
        lambda p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    raw, faiss = lib_dirs
    chat_service.answer_user_query(
        "explain why machine learning works in detail for my documents",
        raw_dir=raw,
        faiss_folder=faiss,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )
    assert routes == ["general_retrieval_failed"]


def _chunk(text: str = "alpha beta gamma", dist: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        rank=0,
        page_content=text,
        metadata={"source_name": "a.txt", "chunk_id": "c1", "rrf_score": 0.02},
        distance=dist,
        score_kind="faiss_l2",
    )


@patch("app.services.chat_service.generate_grounded_answer")
@patch("app.services.chat_service.prepare_web_for_generation")
@patch("app.services.chat_service.select_generation_context")
@patch("app.services.chat_service.document_health.allow_document_grounding", return_value=True)
@patch("app.services.chat_service.document_health.filter_trusted_retrieval_hits", side_effect=lambda f, h: h)
@patch("app.services.chat_service.rerank_hybrid_hits", side_effect=lambda h: h)
@patch("app.services.chat_service.hybrid_retrieve", return_value=[MagicMock()])
@patch("app.services.chat_service.rewrite_for_retrieval", return_value="rewritten")
@patch("app.services.chat_service.faiss_vector_count", return_value=5)
def test_grounded_when_docs_good(
    _fv,
    _rw,
    _hr,
    _rr,
    _ft,
    _adg,
    _sgc,
    pweb,
    gga,
    lib_dirs,
    monkeypatch,
):
    routes = _capture_routing(monkeypatch)
    pweb.return_value = ([], "(No web results.)", [], "")
    _sgc.return_value = [_chunk()]
    from app.llm.generator import GroundedAnswer, SourceRef

    gga.return_value = GroundedAnswer(
        answer="Answer with [SOURCE 1].",
        sources=(
            SourceRef(
                source_number=1,
                chunk_id="c1",
                source_name="a.txt",
                page_label="-",
                file_path="",
            ),
        ),
    )
    (lib_dirs[0] / "a.txt").write_text("x", encoding="utf-8")

    def _list(rd: Path):
        return sorted(rd.glob("*"))

    monkeypatch.setattr(chat_service.index_service, "list_raw_files", _list)
    monkeypatch.setattr(
        chat_service.index_service,
        "ensure_index_matches_library",
        lambda *a, **k: (True, ""),
    )
    monkeypatch.setattr(chat_service, "wants_no_retrieval_fastpath", lambda q: False)
    monkeypatch.setattr(chat_service.index_service, "load_faiss_store", lambda p: object())
    monkeypatch.setattr(chat_service, "_streaming_enabled", lambda: False)
    raw, faiss = lib_dirs
    chat_service.answer_user_query(
        "explain why machine learning works in detail for my documents",
        raw_dir=raw,
        faiss_folder=faiss,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )
    assert routes == ["grounded"]
    gga.assert_called_once()


@patch("app.services.chat_service.generate_web_grounded_answer", return_value="See [t](http://u)")
@patch("app.services.chat_service.validate_web_markdown_links", return_value=("fixed", None))
@patch("app.services.chat_service.web_results_strong_enough", return_value=True)
@patch("app.services.chat_service.prepare_web_for_generation")
@patch("app.services.chat_service.select_generation_context", return_value=[])
@patch("app.services.chat_service.document_health.allow_document_grounding", return_value=False)
@patch("app.services.chat_service.document_health.filter_trusted_retrieval_hits", side_effect=lambda f, h: h)
@patch("app.services.chat_service.rerank_hybrid_hits", side_effect=lambda h: h)
@patch("app.services.chat_service.hybrid_retrieve", return_value=[MagicMock()])
@patch("app.services.chat_service.rewrite_for_retrieval", return_value="rewritten")
@patch("app.services.chat_service.faiss_vector_count", return_value=5)
def test_web_when_docs_weak_web_strong(
    _fv,
    _rw,
    _hr,
    _rr,
    _ft,
    _adg,
    _sgc,
    pweb,
    _wrs,
    _vwm,
    _gwa,
    lib_dirs,
    monkeypatch,
):
    from app.services.web_search_service import WebSnippet

    routes = _capture_routing(monkeypatch)
    snip = WebSnippet(title="T", url="http://example.com/a", snippet="x" * 60)
    pweb.return_value = ([snip], "[WEB 1]", [{"url": snip.url}], "shaped")
    (lib_dirs[0] / "a.txt").write_text("x", encoding="utf-8")

    def _list(rd: Path):
        return sorted(rd.glob("*"))

    monkeypatch.setattr(chat_service.index_service, "list_raw_files", _list)
    monkeypatch.setattr(
        chat_service.index_service,
        "ensure_index_matches_library",
        lambda *a, **k: (True, ""),
    )
    monkeypatch.setattr(chat_service, "wants_no_retrieval_fastpath", lambda q: False)
    monkeypatch.setattr(chat_service.index_service, "load_faiss_store", lambda p: object())
    monkeypatch.setattr(chat_service, "_streaming_enabled", lambda: False)
    raw, faiss = lib_dirs
    t = chat_service.answer_user_query(
        "explain why machine learning works in detail for my documents",
        raw_dir=raw,
        faiss_folder=faiss,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )
    assert routes == ["web_weak_docs"]
    assert t.mode == "web"


@patch("app.services.chat_service.generate_blended_answer")
@patch("app.services.chat_service.prepare_web_for_generation")
@patch("app.services.chat_service.select_generation_context")
@patch("app.services.chat_service.document_health.allow_document_grounding", return_value=True)
@patch("app.services.chat_service.document_health.filter_trusted_retrieval_hits", side_effect=lambda f, h: h)
@patch("app.services.chat_service.rerank_hybrid_hits", side_effect=lambda h: h)
@patch("app.services.chat_service.hybrid_retrieve", return_value=[MagicMock()])
@patch("app.services.chat_service.rewrite_for_retrieval", return_value="rewritten")
@patch("app.services.chat_service.faiss_vector_count", return_value=5)
def test_blended_time_sensitive(
    _fv,
    _rw,
    _hr,
    _rr,
    _ft,
    _adg,
    _sgc,
    pweb,
    gba,
    lib_dirs,
    monkeypatch,
):
    from app.llm.generator import GroundedAnswer, SourceRef
    from app.services.web_search_service import WebSnippet

    routes = _capture_routing(monkeypatch)
    snip = WebSnippet(title="News", url="http://news.example/x", snippet="y" * 60)
    pweb.return_value = ([snip], "web block", [{"url": snip.url}], "shaped")
    _sgc.return_value = [_chunk()]
    gba.return_value = GroundedAnswer(
        answer="From doc [SOURCE 1] and [news](http://news.example/x).",
        sources=(
            SourceRef(
                source_number=1,
                chunk_id="c1",
                source_name="a.txt",
                page_label="-",
                file_path="",
            ),
        ),
    )
    (lib_dirs[0] / "a.txt").write_text("x", encoding="utf-8")

    def _list(rd: Path):
        return sorted(rd.glob("*"))

    monkeypatch.setattr(chat_service.index_service, "list_raw_files", _list)
    monkeypatch.setattr(
        chat_service.index_service,
        "ensure_index_matches_library",
        lambda *a, **k: (True, ""),
    )
    monkeypatch.setattr(chat_service, "wants_no_retrieval_fastpath", lambda q: False)
    monkeypatch.setattr(chat_service.index_service, "load_faiss_store", lambda p: object())
    monkeypatch.setattr(chat_service, "web_results_strong_enough", lambda *a, **k: True)
    monkeypatch.setattr(chat_service, "_streaming_enabled", lambda: False)
    raw, faiss = lib_dirs
    chat_service.answer_user_query(
        "What is the latest news today about my uploaded policy?",
        raw_dir=raw,
        faiss_folder=faiss,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )
    assert routes == ["blended"]


@patch("app.services.chat_service.prepare_web_for_generation")
@patch("app.services.chat_service.select_generation_context", return_value=[])
@patch("app.services.chat_service.document_health.allow_document_grounding", return_value=False)
@patch("app.services.chat_service.document_health.filter_trusted_retrieval_hits", side_effect=lambda f, h: h)
@patch("app.services.chat_service.rerank_hybrid_hits", side_effect=lambda h: h)
@patch("app.services.chat_service.hybrid_retrieve", return_value=[MagicMock()])
@patch("app.services.chat_service.rewrite_for_retrieval", return_value="rewritten")
@patch("app.services.chat_service.faiss_vector_count", return_value=5)
def test_general_web_thin_when_web_weak(
    _fv,
    _rw,
    _hr,
    _rr,
    _ft,
    _adg,
    _sgc,
    pweb,
    lib_dirs,
    monkeypatch,
):
    routes = _capture_routing(monkeypatch)
    pweb.return_value = ([MagicMock()], "block", [], "shaped")
    monkeypatch.setattr(chat_service, "safe_general_answer", lambda q: ("g", None))
    monkeypatch.setattr(chat_service, "web_results_strong_enough", lambda *a, **k: False)
    (lib_dirs[0] / "a.txt").write_text("x", encoding="utf-8")

    def _list(rd: Path):
        return sorted(rd.glob("*"))

    monkeypatch.setattr(chat_service.index_service, "list_raw_files", _list)
    monkeypatch.setattr(
        chat_service.index_service,
        "ensure_index_matches_library",
        lambda *a, **k: (True, ""),
    )
    monkeypatch.setattr(chat_service, "wants_no_retrieval_fastpath", lambda q: False)
    monkeypatch.setattr(chat_service.index_service, "load_faiss_store", lambda p: object())
    monkeypatch.setattr(chat_service, "_streaming_enabled", lambda: False)
    raw, faiss = lib_dirs
    chat_service.answer_user_query(
        "explain why machine learning works in detail for my documents",
        raw_dir=raw,
        faiss_folder=faiss,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )
    assert routes == ["general_web_thin"]


def test_phase15_scenarios_json_valid():
    from app.reliability.scenarios import default_scenarios_path, load_scenarios, validate_scenarios

    data = load_scenarios(default_scenarios_path())
    errs = validate_scenarios(data)
    assert not errs, errs
    assert len(data["scenarios"]) >= 15


def test_reliability_log_json_serializes():
    from app.reliability.turn_log import log_reliability_turn, reliability_log_enabled
    from app.services.chat_service import AssistantTurn

    turn = AssistantTurn(mode="general", text="x")
    # Should not raise even when logging disabled
    log_reliability_turn(turn, routing="general_no_library", retrieval_ran=False)
    assert reliability_log_enabled() in (True, False)
