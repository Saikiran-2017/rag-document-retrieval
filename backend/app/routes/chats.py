from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.persistence import chat_store

from backend.app.schemas.common import AppendMessageRequest, MessageOut, SetChatTitleRequest

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("")
def list_chats(limit: int = 40) -> list[dict[str, Any]]:
    chat_store.init_db()
    return chat_store.list_sessions(max(1, min(100, limit)))


@router.post("")
def create_chat(title: str = "New chat") -> dict[str, str]:
    chat_store.init_db()
    sid = chat_store.create_session(title[:200])
    return {"id": sid, "title": title[:200]}


@router.patch("/{session_id}")
def set_chat_title(session_id: str, body: SetChatTitleRequest) -> dict[str, bool]:
    chat_store.init_db()
    if chat_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat_store.set_session_title(session_id, body.title.strip()[:200])
    return {"ok": True}


@router.post("/{session_id}/messages")
def append_message(session_id: str, body: AppendMessageRequest) -> dict[str, bool]:
    chat_store.init_db()
    if chat_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    extra = dict(body.extra) if body.extra else None
    chat_store.append_message(session_id, body.role, body.content, extra)
    return {"ok": True}


@router.get("/{session_id}/messages")
def get_messages(session_id: str) -> list[MessageOut]:
    chat_store.init_db()
    if chat_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    rows = chat_store.load_messages(session_id)
    out: list[MessageOut] = []
    for m in rows:
        extra = {k: v for k, v in m.items() if k not in ("role", "content")}
        out.append(MessageOut(role=str(m.get("role", "")), content=str(m.get("content", "")), extra=extra))
    return out


@router.delete("/{session_id}")
def delete_chat(session_id: str) -> dict[str, bool]:
    chat_store.init_db()
    if chat_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat_store.delete_session(session_id)
    return {"deleted": True}
