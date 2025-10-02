from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.common import SyncRequest, SyncResponse
from backend.app.services import index_service

router = APIRouter(prefix="/sync", tags=["sync"])


@router.post("", response_model=SyncResponse)
def sync_index(
    body: SyncRequest | None = None,
    settings: Settings = Depends(get_settings),
) -> SyncResponse | JSONResponse:
    body = body or SyncRequest()
    cs = body.chunk_size if body.chunk_size is not None else settings.default_chunk_size
    co = body.chunk_overlap if body.chunk_overlap is not None else settings.default_chunk_overlap
    if co >= cs:
        r = SyncResponse(
            ok=False,
            status="failed",
            message="chunk_overlap must be less than chunk_size.",
            sync_action="failed",
        )
        return JSONResponse(status_code=400, content=r.model_dump(mode="json"))

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)

    ok, msg, nvec, action = index_service.rebuild_knowledge_index(
        settings.raw_dir,
        settings.faiss_dir,
        chunk_size=cs,
        chunk_overlap=co,
    )
    fp = list(index_service.library_content_fingerprint(settings.raw_dir))
    pairs = [[a, b] for a, b in fp]

    if not ok:
        r = SyncResponse(
            ok=False,
            status="failed",
            message=msg or "Indexing failed.",
            vector_count=nvec,
            sync_action="failed",
            content_fingerprint=pairs if fp else None,
        )
        return JSONResponse(status_code=500, content=r.model_dump(mode="json"))

    if action == "unchanged":
        user_msg = "No library changes detected. Search index is up to date."
        st = "unchanged"
    else:
        user_msg = msg.strip() if msg else "Library indexed successfully."
        st = "success"

    return SyncResponse(
        ok=True,
        status=st,  # type: ignore[arg-type]
        message=user_msg,
        vector_count=nvec,
        sync_action=action,
        content_fingerprint=pairs if fp else None,
    )
