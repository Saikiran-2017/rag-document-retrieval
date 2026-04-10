from __future__ import annotations

import os
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.schemas.serialization import chat_metadata_dict, chat_response_from_turn
from backend.app.services import chat_service
from backend.app.utils.sse import format_sse

router = APIRouter(prefix="/chat", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("", response_model=ChatResponse)
def post_chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    Non-streaming chat: full JSON body (same routing and validation as Streamlit).

    Sets ``KA_NO_STREAM=1`` for the duration of the call.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    cs = req.chunk_size if req.chunk_size is not None else settings.default_chunk_size
    co = req.chunk_overlap if req.chunk_overlap is not None else settings.default_chunk_overlap
    tk = req.top_k if req.top_k is not None else settings.default_top_k
    if co >= cs:
        raise HTTPException(status_code=400, detail="chunk_overlap must be less than chunk_size")
    prev = os.environ.get("KA_NO_STREAM")
    os.environ["KA_NO_STREAM"] = "1"
    conv = None
    if req.conversation:
        conv = [t.model_dump(exclude_none=True) for t in req.conversation]

    try:
        turn = chat_service.answer_user_query(
            req.message.strip(),
            raw_dir=settings.raw_dir,
            faiss_folder=settings.faiss_dir,
            chunk_size=cs,
            chunk_overlap=co,
            top_k=tk,
            task_mode=req.task_mode,
            summarize_scope=req.summarize_scope,
            conversation_history=conv,
        )
    finally:
        if prev is None:
            os.environ.pop("KA_NO_STREAM", None)
        else:
            os.environ["KA_NO_STREAM"] = prev
    if turn.stream_tokens:
        full = "".join(turn.stream_tokens())
        turn = chat_service.materialize_streamed_turn(turn, full)
    return chat_response_from_turn(turn)


def _chat_sse_events(req: ChatRequest, settings: Settings) -> Iterator[str]:
    """
    SSE sequence: ``start`` (immediate) → ``token`` (zero or more) → ``done`` | ``error``.

    Uses the same ``answer_user_query`` routing as Streamlit; does not persist to SQLite
    (callers can persist using the ``done`` payload).
    """
    yield format_sse({"type": "start"}, event="start")

    if not req.message.strip():
        yield format_sse({"type": "error", "detail": "message is required"}, event="error")
        return

    cs = req.chunk_size if req.chunk_size is not None else settings.default_chunk_size
    co = req.chunk_overlap if req.chunk_overlap is not None else settings.default_chunk_overlap
    tk = req.top_k if req.top_k is not None else settings.default_top_k
    if co >= cs:
        yield format_sse(
            {"type": "error", "detail": "chunk_overlap must be less than chunk_size"},
            event="error",
        )
        return

    prev_no_stream = os.environ.get("KA_NO_STREAM")
    if prev_no_stream is not None:
        os.environ.pop("KA_NO_STREAM", None)

    try:
        try:
            conv = None
            if req.conversation:
                conv = [t.model_dump(exclude_none=True) for t in req.conversation]
            turn = chat_service.answer_user_query(
                req.message.strip(),
                raw_dir=settings.raw_dir,
                faiss_folder=settings.faiss_dir,
                chunk_size=cs,
                chunk_overlap=co,
                top_k=tk,
                task_mode=req.task_mode,
                summarize_scope=req.summarize_scope,
                conversation_history=conv,
            )
        except Exception as exc:
            yield format_sse(
                {"type": "error", "detail": f"{type(exc).__name__}: {exc}"},
                event="error",
            )
            return

        if turn.stream_tokens:
            buf: list[str] = []
            for piece in turn.stream_tokens():
                buf.append(piece)
                if piece:
                    yield format_sse({"type": "token", "delta": piece}, event="token")
            turn = chat_service.materialize_streamed_turn(turn, "".join(buf))
        else:
            chunk = turn.text or turn.error or ""
            if chunk:
                yield format_sse({"type": "token", "delta": chunk}, event="token")

        meta = chat_metadata_dict(turn)
        yield format_sse({"type": "done", "answer": meta}, event="done")
    finally:
        if prev_no_stream is not None:
            os.environ["KA_NO_STREAM"] = prev_no_stream


@router.post("/stream")
def post_chat_stream(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Streaming chat (Server-Sent Events). Same routing as ``POST /api/v1/chat``.

    **Events**

    - ``start``: connection accepted (sent before retrieval / generation).
    - ``token``: ``data.delta`` is a text fragment (document, web, blended, or general).
    - ``done``: ``data.answer`` matches :class:`ChatResponse` JSON (mode, full text, sources, web_snippets, notes, warnings).
    - ``error``: ``data.detail`` describes the failure.

    Temporarily clears ``KA_NO_STREAM`` if set so token iterators are used.
    """
    return StreamingResponse(
        _chat_sse_events(req, settings),
        media_type="text/event-stream; charset=utf-8",
        headers=dict(_SSE_HEADERS),
    )
