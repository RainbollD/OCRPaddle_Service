import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.config import settings
from app.ocr_engine import init_ocr, run_ocr_on_file
from app.pdf_utils import pdf_to_images
from app.schemas import (
    BatchImageResult,
    BatchOCRResponse,
    HealthResponse,
    ImageOCRResponse,
    PDFOCRResponse,
    PageResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising OCR engine...")
    init_ocr()
    yield
    logger.info("Shutting down OCR service.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OCR Service",
    description="Local OCR API backed by PaddleOCR (PaddleX v3).",
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


@app.post("/ocr/image", response_model=ImageOCRResponse, tags=["OCR"])
async def ocr_image(file: UploadFile = File(...)) -> ImageOCRResponse:
    _validate_upload_size(file)
    filename = file.filename or "unknown"
    _validate_image_extension(filename)

    ext = Path(filename).suffix.lower()
    tmp_path = await _save_upload_to_tmp(file, suffix=ext)

    try:
        t0 = time.perf_counter()
        lines, text = await run_ocr_on_file(tmp_path)
        elapsed = time.perf_counter() - t0

        logger.info(
            "image | file=%s ext=%s lines=%d time=%.3fs",
            filename, ext, len(lines), elapsed,
        )
        return ImageOCRResponse(
            filename=filename,
            text=text,
            lines=lines,
            language=settings.ocr_lang,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        _remove_file(tmp_path)


@app.post("/ocr/document", response_model=PDFOCRResponse, tags=["OCR"])
async def ocr_pdf(file: UploadFile = File(...)) -> PDFOCRResponse:
    _validate_upload_size(file)
    filename = file.filename or "unknown"

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

        page_results: list[PageResult] = []
        for idx, img_path in enumerate(page_image_paths, start=1):
            try:
                lines, text = await run_ocr_on_file(img_path)
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"OCR failed on page {idx}: {exc}",
                ) from exc
            page_results.append(PageResult(page=idx, text=text, lines=lines))

        full_text = "\n\n".join(p.text for p in page_results)
        total_lines = sum(len(p.lines) for p in page_results)
        elapsed = time.perf_counter() - t0

        logger.info(
            "pdf | file=%s pages=%d lines=%d time=%.3fs",
            filename, len(page_results), total_lines, elapsed,
        )
        return PDFOCRResponse(
            filename=filename,
            pages=page_results,
            full_text=full_text,
        )
    finally:
        _remove_file(tmp_pdf)
        for p in page_image_paths:
            _remove_file(p)


@app.post("/ocr/batch", response_model=BatchOCRResponse, tags=["OCR"])
async def ocr_batch(files: list[UploadFile] = File(...)) -> BatchOCRResponse:
    results: list[BatchImageResult] = []

    for file in files:
        filename = file.filename or "unknown"
        ext = Path(filename).suffix.lower()
        # Per-file size check
        if file.size and file.size > settings.max_upload_bytes:
            results.append(
                BatchImageResult(
                    filename=filename,
                    text="",
                    lines=[],
                    error=f"File exceeds {settings.max_upload_mb} MB limit.",
                )
            )
            continue

        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            results.append(
                BatchImageResult(
                    filename=filename,
                    text="",
                    lines=[],
                    error=f"Unsupported extension '{ext}'.",
                )
            )
            continue

        try:
            tmp_path = await _save_upload_to_tmp(file, suffix=ext)
        except HTTPException as exc:
            results.append(
                BatchImageResult(
                    filename=filename,
                    text="",
                    lines=[],
                    error=str(exc.detail),
                )
            )
            continue

        try:
            t0 = time.perf_counter()
            lines, text = await run_ocr_on_file(tmp_path)
            elapsed = time.perf_counter() - t0
            logger.info(
                "batch | file=%s ext=%s lines=%d time=%.3fs",
                filename, ext, len(lines), elapsed,
            )
            results.append(
                BatchImageResult(
                    filename=filename,
                    text=text,
                    lines=lines,
                    language=settings.ocr_lang,
                )
            )
        except RuntimeError as exc:
            results.append(
                BatchImageResult(
                    filename=filename,
                    text="",
                    lines=[],
                    error=str(exc),
                )
            )
        finally:
            _remove_file(tmp_path)

    return BatchOCRResponse(results=results)
