"""Save uploaded files to the raw library directory."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SUPPORTED_EXT = {".pdf", ".docx", ".txt"}

# Default cap (bytes); override with KA_MAX_UPLOAD_BYTES.
_DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Windows reserved device names (stem without extension).
_WIN_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def max_upload_bytes() -> int:
    """
    Load max upload size from environment or use default.
    
    Enforces bounds: 1KB minimum, 512GB maximum.
    Set KA_MAX_UPLOAD_BYTES to override (bytes).
    """
    raw = os.environ.get("KA_MAX_UPLOAD_BYTES", "").strip()
    if raw.isdigit():
        return max(1024, min(512 * 1024 * 1024, int(raw)))
    return _DEFAULT_MAX_UPLOAD_BYTES


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes data for integrity verification."""
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    """Compute SHA256 hash of file at path (memory-efficient for large files)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sanitize_upload_basename(name: str) -> tuple[str | None, str | None]:
    """
    Validate and sanitize uploaded filename.
    
    Return (safe_basename, reject_reason).
    
    Rejects:
    - Path traversal attempts (../../, null bytes)
    - Empty or whitespace-only names
    - Control characters (ASCII < 32)
    - Windows reserved device names (CON, PRN, NUL, LPT1-9, COM1-9)
    - Special directory markers (., ..)
    
    See https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file
    """
    if not name or not name.strip():
        return None, "empty_filename"
    base = Path(name.replace("\x00", "")).name.strip()
    if not base or base in (".", ".."):
        return None, "invalid_filename"
    if any(ord(c) < 32 for c in base):
        return None, "invalid_filename"
    stem = Path(base).stem.lower()
    if stem in _WIN_RESERVED:
        return None, "reserved_filename"
    if ".." in base or "/" in base or "\\" in base:
        return None, "invalid_filename"
    return base, None


@dataclass(frozen=True)
class UploadSaveItem:
    original_filename: str
    stored_filename: str
    status: Literal["saved", "duplicate_unchanged", "rejected"]
    detail: str | None = None
    sha256_hex: str | None = None


def save_upload_batch(
    uploaded_files: list[Any] | None,
    raw_dir: Path,
    *,
    enforce_supported_extensions: bool = False,
    max_bytes: int | None = None,
) -> list[UploadSaveItem]:
    """
    Write uploads to ``raw_dir``.

    When ``enforce_supported_extensions`` is True, only :data:`SUPPORTED_EXT` are accepted.
    Otherwise (Streamlit path) extension is not enforced here; indexing still filters by type.

    Duplicate policy: same target path and identical SHA-256 as existing file →
    ``duplicate_unchanged`` (no write). New or changed bytes → ``saved``.
    """
    if not uploaded_files:
        return []

    cap = max_bytes if max_bytes is not None else max_upload_bytes()
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_resolved = raw_dir.resolve()

    results: list[UploadSaveItem] = []
    for f in uploaded_files:
        original = str(getattr(f, "name", "") or "upload").strip()
        safe, reason = sanitize_upload_basename(original)
        if not safe:
            results.append(
                UploadSaveItem(
                    original_filename=original or "(unnamed)",
                    stored_filename=Path(original).name if original else "",
                    status="rejected",
                    detail=reason or "invalid_filename",
                )
            )
            continue

        suf = "." + safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
        if enforce_supported_extensions and suf not in SUPPORTED_EXT:
            results.append(
                UploadSaveItem(
                    original_filename=original,
                    stored_filename=safe,
                    status="rejected",
                    detail="unsupported_file_type",
                    sha256_hex=None,
                )
            )
            continue

        try:
            data = f.getvalue()
        except Exception:
            results.append(
                UploadSaveItem(
                    original_filename=original,
                    stored_filename=safe,
                    status="rejected",
                    detail="read_failed",
                )
            )
            continue

        if not data:
            results.append(
                UploadSaveItem(
                    original_filename=original,
                    stored_filename=safe,
                    status="rejected",
                    detail="empty_file",
                )
            )
            continue

        if len(data) > cap:
            results.append(
                UploadSaveItem(
                    original_filename=original,
                    stored_filename=safe,
                    status="rejected",
                    detail="file_too_large",
                )
            )
            continue

        digest = _sha256_bytes(data)
        dest = raw_dir / safe
        try:
            dest_resolved = dest.resolve()
            dest_resolved.relative_to(raw_resolved)
        except (OSError, ValueError):
            results.append(
                UploadSaveItem(
                    original_filename=original,
                    stored_filename=safe,
                    status="rejected",
                    detail="invalid_target_path",
                )
            )
            continue

        if dest.is_file():
            try:
                if _sha256_path(dest) == digest:
                    results.append(
                        UploadSaveItem(
                            original_filename=original,
                            stored_filename=safe,
                            status="duplicate_unchanged",
                            detail=None,
                            sha256_hex=digest,
                        )
                    )
                    continue
            except OSError:
                pass

        dest.write_bytes(data)
        results.append(
            UploadSaveItem(
                original_filename=original,
                stored_filename=safe,
                status="saved",
                detail=None,
                sha256_hex=digest,
            )
        )

    return results


def save_uploads_to_raw(uploaded_files: list[Any] | None, raw_dir: Path) -> tuple[int, list[str]]:
    """
    Write each uploaded file to raw_dir; returns (count_saved, filenames).

    ``count_saved`` counts only rows where bytes were written (not duplicate skips or rejects).
    ``filenames`` lists ``stored_filename`` for every non-rejected item (saved + duplicate).
    """
    if not uploaded_files:
        return 0, []
    items = save_upload_batch(
        uploaded_files,
        raw_dir,
        enforce_supported_extensions=False,
        max_bytes=max_upload_bytes(),
    )
    n_saved = sum(1 for it in items if it.status == "saved")
    names = [it.stored_filename for it in items if it.status in ("saved", "duplicate_unchanged")]
    return n_saved, names
