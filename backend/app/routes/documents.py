from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.library_catalog import list_document_catalog

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.common import DocumentRow, DocumentsListResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentsListResponse)
def list_documents(settings: Settings = Depends(get_settings)) -> DocumentsListResponse:
    """
    Library catalog: supported files in the raw folder with manifest health (after Sync).

    Health values: ``uploaded``, ``processing``, ``ready``, ``ready_limited``, ``failed``.
    """
    rows = list_document_catalog(settings.raw_dir, settings.faiss_dir)
    docs = [DocumentRow(**r) for r in rows]
    return DocumentsListResponse(documents=docs, count=len(docs))
