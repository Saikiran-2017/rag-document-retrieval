"""
Per-file document status for the FAISS library (upload → index → retrieval probe).

Phase 13: strict readiness (parse → chunks → index counts → probe), health states,
and plain-language notes for the UI. Internal diagnostics stay out of user-facing strings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from app.services import message_service as msg

FileHealthStatus = Literal["uploaded", "processing", "ready", "ready_limited", "failed"]

_MANIFEST_NAME = "document_manifest.json"


def _manifest_path(faiss_folder: Path) -> Path:
    return Path(faiss_folder) / _MANIFEST_NAME


def load_manifest(faiss_folder: Path) -> dict[str, Any]:
    """Return manifest payload for UI (files keyed by filename)."""
    data = _load(faiss_folder)
    return {"version": data.get("version", 1), "files": dict(data.get("files") or {})}


def _load(faiss_folder: Path) -> dict[str, Any]:
    p = _manifest_path(faiss_folder)
    if not p.is_file():
        return {"version": 1, "files": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "files": {}}
    if not isinstance(data, dict):
        return {"version": 1, "files": {}}
    files = data.get("files")
    if not isinstance(files, dict):
        data["files"] = {}
    return data


def _save(faiss_folder: Path, data: dict[str, Any]) -> None:
    p = _manifest_path(faiss_folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _friendly_failure(reason: str | None) -> str:
    if reason == "parse":
        return msg.DOC_HEALTH_COULD_NOT_READ_FILE
    if reason in ("empty_ingest", "no_chunks"):
        return msg.DOC_HEALTH_NO_TEXT_EXTRACTED
    return msg.DOC_HEALTH_GENERIC_FAILURE


def set_all_processing(faiss_folder: Path, filenames: list[str]) -> None:
    """Mark listed files as processing before a rebuild attempt."""
    data = _load(faiss_folder)
    files: dict[str, Any] = data.setdefault("files", {})
    now = time.time()
    for name in filenames:
        row = files.get(name)
        if not isinstance(row, dict):
            row = {}
        row["status"] = "processing"
        row["updated_at"] = now
        row.pop("user_facing_note", None)
        row.pop("internal_error", None)
        row.pop("parse_ok", None)
        row.pop("chunk_count_expected", None)
        row.pop("chunk_count_indexed", None)
        row.pop("retrieval_probe_ok", None)
        files[name] = row
    _save(faiss_folder, data)


def mark_index_failure(faiss_folder: Path, message: str) -> None:
    """On catastrophic index failure, mark any processing rows as failed with calm copy."""
    data = _load(faiss_folder)
    files: dict[str, Any] = data.setdefault("files", {})
    now = time.time()
    for name, row in list(files.items()):
        if not isinstance(row, dict):
            continue
        if row.get("status") == "processing":
            row["status"] = "failed"
            row["user_facing_note"] = msg.DOC_HEALTH_INDEX_INTERRUPTED
            row["internal_error"] = (message or "")[:500]
            row["updated_at"] = now
            files[name] = row
    _save(faiss_folder, data)


def _probe_ok(probe: dict[str, Any] | None) -> bool:
    if not probe or not isinstance(probe, dict):
        return False
    return bool(probe.get("ok"))


def apply_post_index_validation(
    faiss_folder: Path,
    *,
    file_build: dict[str, dict[str, Any]],
    actual_chunk_counts: dict[str, int],
    retrieval_probe: dict[str, Any] | None,
) -> None:
    """
    After FAISS build + probe: set per-file status from parse/chunk/index/probe.

    file_build[name]: { ok: bool, reason?: str, expected_chunks: int, reused?: bool }
    actual_chunk_counts: vectors per source_name in the store.
    """
    data = _load(faiss_folder)
    files: dict[str, Any] = data.setdefault("files", {})
    now = time.time()
    probe_passed = _probe_ok(retrieval_probe)

    for name, fb in file_build.items():
        if not isinstance(fb, dict):
            continue
        row = files.get(name)
        if not isinstance(row, dict):
            row = {}
        ok = bool(fb.get("ok", False))
        expected = int(fb.get("expected_chunks") or 0)
        actual = int(actual_chunk_counts.get(name, 0))

        row["parse_ok"] = ok
        row["chunk_count_expected"] = expected
        row["chunk_count_indexed"] = actual
        row["retrieval_probe_ok"] = probe_passed
        row["updated_at"] = now

        if not ok:
            reason = str(fb.get("reason") or "")
            row["status"] = "failed"
            row["user_facing_note"] = _friendly_failure(reason or None)
            row["internal_error"] = str(fb.get("internal") or "")[:500] or None
            files[name] = row
            continue

        if expected > 0 and actual != expected:
            row["status"] = "ready_limited"
            row["user_facing_note"] = msg.DOC_HEALTH_INDEX_SECTIONS_MISMATCH
            row.pop("internal_error", None)
            files[name] = row
            continue

        if not probe_passed:
            row["status"] = "ready_limited"
            row["user_facing_note"] = msg.DOC_HEALTH_SEARCH_PARTIALLY_RELIABLE
            row.pop("internal_error", None)
            files[name] = row
            continue

        row["status"] = "ready"
        row.pop("user_facing_note", None)
        row.pop("internal_error", None)
        files[name] = row

    _save(faiss_folder, data)


def apply_probe_only_refresh(
    faiss_folder: Path,
    *,
    filenames: list[str],
    file_chunk_counts: dict[str, int],
    actual_chunk_counts: dict[str, int],
    retrieval_probe: dict[str, Any] | None,
) -> None:
    """When index was skipped: re-check counts + probe and adjust ready / ready_limited."""
    data = _load(faiss_folder)
    files: dict[str, Any] = data.setdefault("files", {})
    now = time.time()
    probe_passed = _probe_ok(retrieval_probe)

    for name in filenames:
        row = files.get(name)
        if not isinstance(row, dict):
            row = {"status": "uploaded"}
        expected = int(file_chunk_counts.get(name, row.get("chunk_count_expected") or 0) or 0)
        actual = int(actual_chunk_counts.get(name, 0))
        row["chunk_count_expected"] = expected
        row["chunk_count_indexed"] = actual
        row["retrieval_probe_ok"] = probe_passed
        row["updated_at"] = now

        prev = str(row.get("status") or "uploaded")
        if prev == "failed":
            files[name] = row
            continue

        if expected > 0 and actual != expected:
            row["status"] = "ready_limited"
            row["user_facing_note"] = msg.DOC_HEALTH_INDEX_SECTIONS_MISMATCH
        elif not probe_passed:
            row["status"] = "ready_limited"
            row["user_facing_note"] = msg.DOC_HEALTH_SEARCH_PARTIALLY_RELIABLE
        else:
            row["status"] = "ready"
            row.pop("user_facing_note", None)
        row.pop("internal_error", None)
        files[name] = row

    _save(faiss_folder, data)


def file_health_status(faiss_folder: Path, source_name: str) -> FileHealthStatus:
    data = _load(faiss_folder)
    row = data.get("files", {}).get(source_name)
    if not isinstance(row, dict):
        return "uploaded"
    s = row.get("status")
    if s in ("uploaded", "processing", "ready", "ready_limited", "failed"):
        return s  # type: ignore[return-value]
    return "uploaded"


def library_has_no_fully_healthy_file(faiss_folder: Path) -> bool:
    """True if there is no file in manifest with status ready."""
    data = _load(faiss_folder)
    files = data.get("files", {})
    if not isinstance(files, dict) or not files:
        return True
    for row in files.values():
        if isinstance(row, dict) and row.get("status") == "ready":
            return False
    return True


def library_health_counts(faiss_folder: Path) -> dict[str, int]:
    data = _load(faiss_folder)
    files = data.get("files", {})
    out = {"ready": 0, "ready_limited": 0, "failed": 0, "processing": 0, "uploaded": 0, "other": 0}
    if not isinstance(files, dict):
        return out
    for row in files.values():
        if not isinstance(row, dict):
            out["other"] += 1
            continue
        st = str(row.get("status") or "uploaded")
        if st in out:
            out[st] += 1
        else:
            out["other"] += 1
    return out


def user_facing_note_for_file(faiss_folder: Path, source_name: str) -> str | None:
    data = _load(faiss_folder)
    row = data.get("files", {}).get(source_name)
    if not isinstance(row, dict):
        return None
    n = row.get("user_facing_note")
    return str(n).strip() if n else None


def list_files_with_status(faiss_folder: Path) -> list[dict[str, Any]]:
    data = _load(faiss_folder)
    files = data.get("files", {})
    if not isinstance(files, dict):
        return []
    out: list[dict[str, Any]] = []
    for name, row in sorted(files.items()):
        if isinstance(row, dict):
            out.append({"name": name, **row})
        else:
            out.append({"name": name, "status": "uploaded"})
    return out


def file_status(faiss_folder: Path, source_name: str) -> dict[str, Any] | None:
    data = _load(faiss_folder)
    row = data.get("files", {}).get(source_name)
    if isinstance(row, dict):
        return dict(row)
    return None

