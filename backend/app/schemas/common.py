from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class UploadFileOutcome(BaseModel):
    original_name: str
    stored_name: str
    status: Literal["saved", "duplicate_unchanged", "rejected"]
    message: str | None = None
    sha256: str | None = None


class UploadResponse(BaseModel):
    """Result of ``POST /api/v1/upload`` (multipart files)."""

    ok: bool
    status: Literal["success", "partial_success", "no_op", "failed"]
    message: str
    files: list[UploadFileOutcome] = Field(default_factory=list)
    saved_count: int = 0
    duplicate_count: int = 0
    rejected_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def filenames(self) -> list[str]:
        """Stored names for non-rejected items (saved + duplicate), for backward compatibility."""
        return [f.stored_name for f in self.files if f.status != "rejected"]


class SyncRequest(BaseModel):
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class SyncResponse(BaseModel):
    ok: bool
    status: Literal["success", "unchanged", "failed"]
    message: str = ""
    vector_count: int = 0
    """``unchanged`` | ``rebuilt`` | ``failed``; mirrors :func:`app.services.index_service.rebuild_knowledge_index` action."""
    sync_action: str = "failed"
    content_fingerprint: list[list[str]] | None = None


class DocumentRow(BaseModel):
    filename: str
    health: str
    note: str | None = None
    updated_at: str | None = None


class DocumentsListResponse(BaseModel):
    documents: list[DocumentRow] = Field(default_factory=list)
    count: int = 0


class MessageOut(BaseModel):
    role: str
    content: str
    extra: dict = Field(default_factory=dict)


class AppendMessageRequest(BaseModel):
    role: str
    content: str
    extra: dict = Field(default_factory=dict)


class SetChatTitleRequest(BaseModel):
    title: str
