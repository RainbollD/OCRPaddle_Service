import logging
import os
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
ALLOWED_OCR_DEVICES = {"cpu", "gpu:0"}


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
    if file.size and file.size > settings.max_upload_bytes:
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


def _resolve_ocr_device(device: str | None) -> str:
    if device is None or not device.strip():
        return settings.ocr_device

    normalized = device.strip().lower()
    if normalized not in ALLOWED_OCR_DEVICES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported OCR device '{device}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_OCR_DEVICES))}"
            ),
        )

    return normalized


async def _save_upload_to_tmp(file: UploadFile, suffix: str) -> Path:
    """Persist the uploaded SpooledTemporaryFile to a real on-disk temp file."""
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded file '{file.filename}' is empty.",
        )
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File '{file.filename}' exceeds the maximum allowed size "
                f"of {settings.max_upload_mb} MB."
            ),
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


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
            page_image_paths = pdf_to_images(tmp_pdf)
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

