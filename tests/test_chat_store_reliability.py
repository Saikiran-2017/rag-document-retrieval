"""Phase 15: SQLite chat persistence (isolated DB path)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_chat_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbf = tmp_path / "chat_phase15.db"
    monkeypatch.setattr("app.persistence.chat_store.db_path", lambda: dbf)
    from app.persistence import chat_store

    chat_store.init_db()
    return chat_store


def test_session_create_and_messages_roundtrip(isolated_chat_db):
    cs = isolated_chat_db
    sid = cs.create_session("Roundtrip")
    cs.append_message(sid, "user", "Hello", None)
    cs.append_message(sid, "assistant", "Hi there", {"grounded": False, "web_sources": []})
    msgs = cs.load_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "Hello"
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"] == "Hi there"
    assert msgs[1].get("grounded") is False


def test_two_sessions_no_cross_contamination(isolated_chat_db):
    cs = isolated_chat_db
    a = cs.create_session("A")
    b = cs.create_session("B")
    cs.append_message(a, "user", "only A", None)
    cs.append_message(b, "user", "only B", None)
    assert len(cs.load_messages(a)) == 1
    assert len(cs.load_messages(b)) == 1
    assert cs.load_messages(a)[0]["content"] == "only A"
    assert cs.load_messages(b)[0]["content"] == "only B"


def test_delete_session_removes_messages(isolated_chat_db):
    cs = isolated_chat_db
    sid = cs.create_session("Del")
    cs.append_message(sid, "user", "x", None)
    cs.delete_session(sid)
    assert cs.get_session(sid) is None
    assert cs.load_messages(sid) == []
