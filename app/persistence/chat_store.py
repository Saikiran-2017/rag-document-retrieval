"""SQLite persistence for chat sessions (GPT-style history in sidebar)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def db_path() -> Path:
    p = _project_root() / "data" / "chat_history.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(db_path()), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                extra_json TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
            )
            """
        )
        c.commit()


def create_session(title: str = "New chat") -> str:
    init_db()
    sid = str(uuid.uuid4())
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, title[:200], now, now),
        )
        c.commit()
    return sid


def list_sessions(limit: int = 40) -> list[dict[str, Any]]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, title, updated_at FROM chat_sessions
            ORDER BY updated_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict[str, Any] | None:
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT id, title, updated_at FROM chat_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def touch_session(session_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        c.commit()


def set_session_title(session_id: str, title: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title[:200], time.time(), session_id),
        )
        c.commit()


def append_message(session_id: str, role: str, content: str, extra: dict[str, Any] | None = None) -> None:
    init_db()
    ex = json.dumps(extra) if extra else None
    with _conn() as c:
        c.execute(
            "INSERT INTO chat_messages (session_id, role, content, extra_json) VALUES (?,?,?,?)",
            (session_id, role, content, ex),
        )
        c.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        c.commit()


def load_messages(session_id: str) -> list[dict[str, Any]]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            """
            SELECT role, content, extra_json FROM chat_messages
            WHERE session_id = ? ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        msg: dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["extra_json"]:
            try:
                msg.update(json.loads(r["extra_json"]))
            except json.JSONDecodeError:
                pass
        out.append(msg)
    return out


def delete_session(session_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        c.commit()
