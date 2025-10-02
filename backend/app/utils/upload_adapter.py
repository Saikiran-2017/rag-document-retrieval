"""Adapt FastAPI ``UploadFile`` to the ``upload_service.save_uploads_to_raw`` duck-typed interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UploadFileView:
    """Mimics Streamlit uploaded file: ``.name`` and ``.getvalue()``."""

    name: str
    _data: bytes

    def getvalue(self) -> bytes:
        return self._data
