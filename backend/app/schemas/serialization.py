from __future__ import annotations

from dataclasses import asdict

from app.services.chat_service import AssistantTurn

from backend.app.schemas.chat import ChatResponse, SourceRefOut


def chat_metadata_dict(turn: AssistantTurn) -> dict:
    """JSON-serializable payload for the final SSE ``done`` event (and clients persisting turns)."""
    return chat_response_from_turn(turn).model_dump(mode="json")


def chat_response_from_turn(turn: AssistantTurn) -> ChatResponse:
    sources = None
    vw = None
    if turn.grounded:
        sources = [SourceRefOut(**asdict(s)) for s in turn.grounded.sources]
        vw = turn.grounded.validation_warning
    rcount = len(turn.hits) if turn.hits else None
    return ChatResponse(
        mode=turn.mode,
        text=(turn.text or "").strip(),
        error=turn.error,
        assistant_note=turn.assistant_note,
        web_snippets=turn.web_snippets,
        sources=sources,
        validation_warning=vw,
        retrieval_chunk_count=rcount,
        diagnostics=turn.diagnostics,
    )
