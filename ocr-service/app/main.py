import asyncio
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.ocr_engine import run_structured_ocr_on_file
from app.pdf_utils import pdf_to_images
from app.schemas import (
    HealthResponse,
    StructuredImageOCRResponse,
    StructuredPageResult,
    StructuredPDFOCRResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Accepts "cpu", "gpu:0", "gpu:1", … — any non-negative integer index.
_GPU_DEVICE_RE = re.compile(r"^gpu:\d+$")

_UPLOAD_CHUNK = 256 * 1024  # 256 KB per read chunk


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up OCR service. OCR engines will be loaded on demand.")
    yield
    logger.info("Shutting down OCR service.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OCR Service",
    description="Local structured OCR API backed by PPStructureV3.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_upload_size(file: UploadFile) -> None:
    """Early-exit if the Content-Length header already reveals an oversized file."""
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File '{file.filename}' exceeds the maximum allowed size "
                f"of {settings.max_upload_mb} MB."
            ),
        )


def _validate_image_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
            ),
        )


def _validate_image_content_type(file: UploadFile) -> None:
    """Validate the MIME type declared in the multipart header."""
    ct = (file.content_type or "").lower().split(";")[0].strip()
    if ct and ct not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type '{ct}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}"
            ),
        )


def _resolve_ocr_device(device: str | None) -> str:
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
    """Stream the uploaded file to a real on-disk temp file.

    Enforces the size limit incrementally so the server never holds the
    entire payload in RAM before deciding to reject it.
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
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ocr/image/structured", response_model=StructuredImageOCRResponse, tags=["OCR"])
async def ocr_image_structured(
    file: UploadFile = File(...),
    device: str | None = Form(default=None),
) -> StructuredImageOCRResponse:
    """Extract text and tables from an image with layout understanding (PPStructureV3)."""
    _validate_upload_size(file)
    filename = file.filename or "unknown"
    _validate_image_extension(filename)
    _validate_image_content_type(file)
    ocr_device = _resolve_ocr_device(device)

    ext = Path(filename).suffix.lower()
    tmp_path = await _save_upload_to_tmp(file, suffix=ext)

    try:
        t0 = time.perf_counter()
        blocks, full_text, full_markdown = await run_structured_ocr_on_file(
            tmp_path,
            device=ocr_device,
        )
        elapsed = time.perf_counter() - t0

        table_count = sum(1 for b in blocks if b.type == "table")
        logger.info(
            "image/structured | file=%s ext=%s device=%s blocks=%d tables=%d time=%.3fs",
            filename, ext, ocr_device, len(blocks), table_count, elapsed,
        )
        return StructuredImageOCRResponse(
            filename=filename,
            blocks=blocks,
            text=full_text,
            markdown=full_markdown,
            language=settings.ocr_lang,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        _remove_file(tmp_path)


@app.post("/ocr/document/structured", response_model=StructuredPDFOCRResponse, tags=["OCR"])
async def ocr_pdf_structured(
    file: UploadFile = File(...),
    device: str | None = Form(default=None),
) -> StructuredPDFOCRResponse:
    """Extract text and tables from a PDF with layout understanding (PPStructureV3)."""
    _validate_upload_size(file)
    filename = file.filename or "unknown"
    ocr_device = _resolve_ocr_device(device)

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted at this endpoint.",
        )

    tmp_pdf = await _save_upload_to_tmp(file, suffix=".pdf")
    page_image_paths: list[Path] = []

    try:
        t0 = time.perf_counter()

        try:
            page_image_paths = await asyncio.to_thread(
                pdf_to_images, tmp_pdf, None, settings.max_pdf_pages
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("PDF rendering failed for %s", filename)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to render PDF: {exc}",
            ) from exc

        page_results: list[StructuredPageResult] = []
        for idx, img_path in enumerate(page_image_paths, start=1):
            try:
                blocks, page_text, page_markdown = await run_structured_ocr_on_file(
                    img_path,
                    device=ocr_device,
                )
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Structured OCR failed on page {idx}: {exc}",
                ) from exc
            page_results.append(
                StructuredPageResult(
                    page=idx,
                    blocks=blocks,
                    text=page_text,
                    markdown=page_markdown,
                )
            )

        full_text = "\n\n".join(p.text for p in page_results)
        full_markdown = "\n\n---\n\n".join(p.markdown for p in page_results)
        elapsed = time.perf_counter() - t0

        total_tables = sum(
            sum(1 for b in p.blocks if b.type == "table") for p in page_results
        )
        logger.info(
            "document/structured | file=%s device=%s pages=%d tables=%d time=%.3fs",
            filename, ocr_device, len(page_results), total_tables, elapsed,
        )
        return StructuredPDFOCRResponse(
            filename=filename,
            pages=page_results,
            full_text=full_text,
            full_markdown=full_markdown,
        )
    finally:
        _remove_file(tmp_pdf)
        for p in page_image_paths:
            _remove_file(p)
