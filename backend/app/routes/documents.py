from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.library_catalog import list_document_catalog
from app.services.library_delete import delete_library_document

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


@router.delete("")
def delete_document(
    filename: str = Query(..., min_length=1, max_length=512),
    settings: Settings = Depends(get_settings),
) -> dict[str, str | bool]:
    """
    Remove one file from the raw library and rebuild the index (or clear index if library is empty).
    """
    ok, msg = delete_library_document(
        settings.raw_dir,
        settings.faiss_dir,
        filename,
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
    )
    if not ok:
        detail = msg or "delete_failed"
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="File not found")
        raise HTTPException(status_code=400, detail=detail)
    return {"ok": True, "message": msg}
