from __future__ import annotations

from fastapi import APIRouter

from app.env_loader import describe_openai_key_for_diagnostics
from app.services.web_search_service import web_search_enabled

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    diag = describe_openai_key_for_diagnostics()
    return {
        "status": "ok",
        "openai_key_configured": bool(diag.get("effective_key_present")),
        "openai_key_placeholder": bool(diag.get("is_placeholder_template")),
        "openai_key_source": diag.get("inferred_value_source"),
        "web_search_enabled": bool(web_search_enabled()),
    }


@router.get("/health/ready")
def ready() -> dict[str, str]:
    """Process is up; optional deeper checks can be added later."""
    return {"status": "ready"}
