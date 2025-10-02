"""Re-exports document trust gates used during retrieval routing."""

from app.services.document_health import (
    allow_document_grounding,
    filter_trusted_retrieval_hits,
)

__all__ = ["allow_document_grounding", "filter_trusted_retrieval_hits"]
