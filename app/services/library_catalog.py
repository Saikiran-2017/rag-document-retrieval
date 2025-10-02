"""Merge on-disk raw library files with FAISS-side manifest for API / UI listing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.persistence import document_manifest
from app.services.index_service import list_raw_files

FileHealthStatus = Literal["uploaded", "processing", "ready", "ready_limited", "failed"]

_VALID: frozenset[str] = frozenset({"uploaded", "processing", "ready", "ready_limited", "failed"})


def _iso(ts: float | int | None) -> str | None:
    if ts is None:
        return None
    try:
        t = float(ts)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def list_document_catalog(raw_dir: Path, faiss_folder: Path) -> list[dict[str, Any]]:
    """
    One row per supported file in ``raw_dir``.

    Health comes from ``document_manifest`` when present; otherwise ``uploaded``.
    ``updated_at`` prefers manifest ``updated_at``, else file mtime.
    """
    manifest = document_manifest.load_manifest(faiss_folder)
    files_meta: dict[str, Any] = dict(manifest.get("files") or {})
    rows: list[dict[str, Any]] = []

    for path in list_raw_files(raw_dir):
        name = path.name
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None

        row = files_meta.get(name)
        note: str | None = None
        health: FileHealthStatus = "uploaded"
        updated_ts: float | int | None = mtime

        if isinstance(row, dict):
            st = row.get("status")
            if isinstance(st, str) and st in _VALID:
                health = st  # type: ignore[assignment]
            n = row.get("user_facing_note")
            if n:
                note = str(n).strip() or None
            u = row.get("updated_at")
            if u is not None:
                try:
                    updated_ts = float(u)
                except (TypeError, ValueError):
                    pass

        rows.append(
            {
                "filename": name,
                "health": health,
                "note": note,
                "updated_at": _iso(updated_ts),
            }
        )

    rows.sort(key=lambda r: r["filename"].lower())
    return rows
