from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    top_k: int | None = None
    task_mode: Literal["auto", "summarize", "extract", "compare"] = "auto"
    summarize_scope: str = "all"


class SourceRefOut(BaseModel):
    source_number: int
    chunk_id: str
    source_name: str
    page_label: str
    file_path: str


class ChatResponse(BaseModel):
    mode: str
    text: str
    error: str | None = None
    assistant_note: str | None = None
    web_snippets: list[dict[str, str]] | None = None
    sources: list[SourceRefOut] | None = None
    validation_warning: str | None = None
    retrieval_chunk_count: int | None = None
    diagnostics: dict[str, Any] | None = None
