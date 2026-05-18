import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.ocr_engine import run_structured_ocr_on_file
from app.pdf_utils import pdf_to_images
from app.schemas import (
    HealthResponse,
    StructuredImageOCRResponse,
    StructuredPageResult,
    StructuredPDFOCRResponse,
)
from app.utils import (
    _remove_file,
    _resolve_ocr_device,
    _save_upload_to_tmp,
    _validate_image_content_type,
    _validate_image_extension,
    _validate_upload_size,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/ocr/image/structured",
    response_model=StructuredImageOCRResponse,
    tags=["OCR"],
)
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


@router.post(
    "/ocr/document/structured",
    response_model=StructuredPDFOCRResponse,
    tags=["OCR"],
)
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
