from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services import debug_service

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/last")
def get_last_diagnostics() -> dict:
    """
    Developer-only snapshot of the last turn's diagnostics.

    Enabled only when ``KA_DEBUG=1`` (or DEVELOPER_DEBUG_MODE).
    """
    if not debug_service.debug_enabled():
        raise HTTPException(status_code=404, detail="debug_disabled")
    d = debug_service.get_last_diagnostics()
    return {"ok": True, "diagnostics": d}

