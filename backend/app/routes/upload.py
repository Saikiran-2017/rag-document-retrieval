from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.upload_service import UploadSaveItem, max_upload_bytes, save_upload_batch

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.common import UploadFileOutcome, UploadResponse
from backend.app.utils.upload_adapter import UploadFileView

router = APIRouter(prefix="/upload", tags=["upload"])

_REJECT_MESSAGES: dict[str, str] = {
    "empty_filename": "Missing or empty filename.",
    "invalid_filename": "Invalid file name.",
    "reserved_filename": "This file name is not allowed on this system.",
    "unsupported_file_type": "Only PDF, DOCX, and TXT files are supported.",
    "read_failed": "Could not read the uploaded file.",
    "empty_file": "The file is empty.",
    "file_too_large": "File exceeds the maximum upload size.",
    "invalid_target_path": "Could not save the file safely.",
}


def _item_to_outcome(item: UploadSaveItem) -> UploadFileOutcome:
    msg: str | None = None
    if item.status == "rejected":
        key = item.detail or ""
        msg = _REJECT_MESSAGES.get(key, key.replace("_", " ").title() if key else "Upload rejected.")
    elif item.status == "duplicate_unchanged":
        msg = "Already in the library with identical content; no changes made."
    return UploadFileOutcome(
        original_name=item.original_filename,
        stored_name=item.stored_filename,
        status=item.status,
        message=msg,
        sha256=item.sha256_hex,
    )


def _build_upload_response(items: list[UploadSaveItem]) -> UploadResponse:
    saved = sum(1 for i in items if i.status == "saved")
    dups = sum(1 for i in items if i.status == "duplicate_unchanged")
    rej = sum(1 for i in items if i.status == "rejected")
    files = [_item_to_outcome(i) for i in items]

    if rej == len(items):
        st: str = "failed"
        ok = False
        message = "No files could be saved. Check types, size limits, and filenames."
    elif saved > 0 and rej == 0:
        st = "success"
        ok = True
        message = (
            "File(s) saved. Run Sync to update the search index."
            if dups == 0
            else "New file(s) saved; duplicate(s) left unchanged. Run Sync to update the search index."
        )
    elif saved > 0 and rej > 0:
        st = "partial_success"
        ok = True
        message = "Some files were saved; others were rejected. See per-file messages."
    elif saved == 0 and dups > 0 and rej == 0:
        st = "no_op"
        ok = True
        message = "No changes: uploaded content already matches the library."
    elif saved == 0 and dups > 0 and rej > 0:
        st = "partial_success"
        ok = True
        message = "Some uploads matched existing files; others were rejected. See per-file messages."
    else:
        st = "failed"
        ok = False
        message = "Upload could not be completed."

    return UploadResponse(
        ok=ok,
        status=st,  # type: ignore[arg-type]
        message=message,
        files=files,
        saved_count=saved,
        duplicate_count=dups,
        rejected_count=rej,
    )


@router.post("", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(),
    settings: Settings = Depends(get_settings),
) -> UploadResponse | JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    views: list[UploadFileView] = []
    for uf in files:
        name = (uf.filename or "upload").strip()
        data = await uf.read()
        views.append(UploadFileView(name=name, _data=data))

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    items = save_upload_batch(
        views,
        settings.raw_dir,
        enforce_supported_extensions=True,
        max_bytes=max_upload_bytes(),
    )
    resp = _build_upload_response(items)
    if resp.status == "failed":
        return JSONResponse(status_code=400, content=resp.model_dump(mode="json"))
    return resp
