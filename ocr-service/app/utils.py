import os
import re
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import settings


_GPU_DEVICE_RE = re.compile(r"^gpu:\d+$")

_UPLOAD_CHUNK = 256 * 1024


def _validate_upload_size(file: UploadFile) -> None:
    """Reject an upload when FastAPI already knows it is too large.

    This check uses the upload metadata before reading the body. The streaming
    save helper enforces the same limit again while bytes are being read.

    Args:
        file (UploadFile): Incoming multipart upload to validate.

    Returns:
        None: The function completes silently when the upload size is allowed.

    Raises:
        HTTPException: Raised with HTTP 413 when the upload size exceeds the
            configured service limit.
    """
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File '{file.filename}' exceeds the maximum allowed size "
                f"of {settings.max_upload_mb} MB."
            ),
        )


def _validate_image_extension(filename: str) -> None:
    """Reject image uploads whose filename extension is not supported.

    Args:
        filename (str): Original filename provided by the client.

    Returns:
        None: The function completes silently when the extension is supported.

    Raises:
        HTTPException: Raised with HTTP 415 when the extension is not listed in
            ``settings.ALLOWED_IMAGE_EXTENSIONS``.
    """
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed: {', '.join(sorted(settings.ALLOWED_IMAGE_EXTENSIONS))}"
            ),
        )


def _resolve_ocr_device(device: str | None) -> str:
    """Validate and normalize the requested PaddleOCR device string.

    Empty values fall back to the configured default. Explicit values must be
    either ``cpu`` or a GPU identifier such as ``gpu:0``.

    Args:
        device (str | None): Optional device value submitted with the request.

    Returns:
        str: Normalized PaddleOCR device string.

    Raises:
        HTTPException: Raised with HTTP 422 when the value is not supported.
    """
    if device is None or not device.strip():
        return settings.ocr_device

    normalized = device.strip().lower()
    if normalized == "cpu" or _GPU_DEVICE_RE.match(normalized):
        return normalized

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            f"Unsupported OCR device '{device}'. "
            "Allowed values: 'cpu', 'gpu:0', 'gpu:1', …"
        ),
    )


async def _save_upload_to_tmp(file: UploadFile, suffix: str) -> Path:
    """Stream an uploaded file to a real on-disk temporary file.

    Enforces the size limit incrementally so the server never holds the
    entire payload in RAM before deciding to reject it.

    Args:
        file (UploadFile): Incoming multipart upload to persist.
        suffix (str): File suffix to use for the temporary file.

    Returns:
        Path: Path to the created temporary file. The caller is responsible for
            deleting it.

    Raises:
        HTTPException: Raised with HTTP 413 when the streamed body exceeds the
            configured size limit.
        HTTPException: Raised with HTTP 422 when the upload body is empty.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    written = 0
    try:
        while chunk := await file.read(_UPLOAD_CHUNK):
            written += len(chunk)
            if written > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File '{file.filename}' exceeds the maximum allowed size "
                        f"of {settings.max_upload_mb} MB."
                    ),
                )
            tmp.write(chunk)
        tmp.flush()
    except Exception:
        tmp.close()
        _remove_file(tmp_path)
        raise
    else:
        tmp.close()

    if written == 0:
        _remove_file(tmp_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded file '{file.filename}' is empty.",
        )
    return tmp_path


def _remove_file(path: Path) -> None:
    """Remove a temporary file if it still exists.

    Missing files and other ``OSError`` failures are ignored because cleanup
    happens in request ``finally`` blocks and should not mask the original
    response or exception.

    Args:
        path (Path): Temporary file path to remove.

    Returns:
        None: This function only performs best-effort cleanup.
    """
    try:
        os.unlink(path)
    except OSError:
        pass
