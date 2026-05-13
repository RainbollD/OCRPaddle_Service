import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from app.config import settings
from paddleocr import PaddleOCR, PPStructureV3

logger = logging.getLogger(__name__)

_ocr_instance: Any | None = None
_ocr_lock = asyncio.Lock()


def _create_ocr_instance() -> Any:
    """Instantiate PaddleOCR with project-configured models."""

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", settings.paddle_pdx_model_source)

    logger.info(
        "Initialising PaddleOCR: det=%s rec=%s lang=%s device=%s",
        settings.text_detection_model,
        settings.text_recognition_model,
        settings.ocr_lang,
        settings.ocr_device,
    )
    return PaddleOCR(
        text_detection_model_name=settings.text_detection_model,
        text_recognition_model_name=settings.text_recognition_model,
        lang=settings.ocr_lang,
        device=settings.ocr_device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _create_structured_ocr_instance() -> Any:
    """Instantiate PPStructureV3 with project-configured models."""

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", settings.paddle_pdx_model_source)

    logger.info(
        "Initialising PPStructureV3: det=%s rec=%s lang=%s device=%s",
        settings.text_detection_model,
        settings.text_recognition_model,
        settings.ocr_lang,
        settings.ocr_device,
    )

    return PPStructureV3(
        text_detection_model_name=settings.text_detection_model,
        text_recognition_model_name=settings.text_recognition_model,
        lang=settings.ocr_lang,
        device=settings.ocr_device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def get_ocr() -> Any:
    """Return the global singleton; raises RuntimeError if not yet initialised."""
    if _ocr_instance is None:
        raise RuntimeError("OCR engine has not been initialised yet.")
    return _ocr_instance


def init_ocr() -> None:
    """Called once at application startup to warm up the model."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = _create_ocr_instance()
        logger.info("PaddleOCR engine ready.")


def init_structured_ocr() -> None:
    """Called once at application startup to warm up the model."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = _create_structured_ocr_instance()
        logger.info("PaddleStructuredOCR engine ready.")


def _as_mapping(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item

    json_data = getattr(item, "json", None)
    if callable(json_data):
        try:
            json_data = json_data()
        except TypeError:
            json_data = None

    if isinstance(json_data, dict):
        nested = json_data.get("res")
        return nested if isinstance(nested, dict) else json_data

    return None


def _lines_from_text(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []

    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_markdown_lines(item: Any) -> list[str]:
    markdown = (
        item.get("markdown") if isinstance(item, dict) else getattr(item, "markdown", None)
    )
    if callable(markdown):
        try:
            markdown = markdown()
        except TypeError:
            markdown = None

    if not isinstance(markdown, dict):
        return []

    # Python PPStructureV3 exposes "markdown_texts"; service-style responses use "text".
    return _lines_from_text(markdown.get("markdown_texts") or markdown.get("text"))


def _extract_rec_texts(mapping: dict[str, Any]) -> list[str]:
    rec_texts = mapping.get("rec_texts")
    if isinstance(rec_texts, list):
        return [str(text).strip() for text in rec_texts if str(text).strip()]

    return []


def _extract_parsing_blocks(mapping: dict[str, Any]) -> list[str]:
    blocks = mapping.get("parsing_res_list")
    if not isinstance(blocks, list):
        return []

    lines: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        lines.extend(_lines_from_text(block.get("block_content")))

    return lines


def _extract_from_mapping(item: Any) -> list[str]:
    markdown_lines = _extract_markdown_lines(item)
    if markdown_lines:
        return markdown_lines

    mapping = _as_mapping(item)
    if mapping is None:
        return []

    nested = mapping.get("res")
    if isinstance(nested, dict):
        mapping = nested

    parsing_lines = _extract_parsing_blocks(mapping)
    if parsing_lines:
        return parsing_lines

    for nested_key in ("overall_ocr_res", "text_paragraphs_ocr_res"):
        nested_result = mapping.get(nested_key)
        if isinstance(nested_result, dict):
            rec_lines = _extract_rec_texts(nested_result)
            if rec_lines:
                return rec_lines

    rec_lines = _extract_rec_texts(mapping)
    if rec_lines:
        return rec_lines

    line_text = mapping.get("transcription") or mapping.get("text")
    if line_text and str(line_text).strip():
        return [str(line_text).strip()]

    return []


def extract_lines_and_text(ocr_result: Any) -> tuple[list[str], str]:
    """Parse raw PaddleOCR output into a list of lines and a joined full text."""
    lines: list[str] = []

    if not ocr_result:
        return lines, ""

    # PaddleOCR v2 returns [[ [box, (text, confidence)], ... ]]
    # PaddleOCR v3/paddlex wraps differently — handle both shapes.
    raw = ocr_result
    if _as_mapping(raw) is not None:
        lines.extend(_extract_from_mapping(raw))
        return lines, "\n".join(lines)

    if isinstance(raw, list) and len(raw) > 0:
        lines.extend(_extract_from_mapping(raw[0]))
        if lines:
            return lines, "\n".join(lines)
        if isinstance(raw[0], list):
            raw = raw[0]

    for item in raw:
        if item is None:
            continue
        try:
            # item: [box, (text, score)]  OR  {"transcription": ..., "text": ...}
            if isinstance(item, dict):
                extracted = _extract_from_mapping(item)
                lines.extend(extracted)
                continue
            else:
                line_text = item[1][0]
            if line_text and line_text.strip():
                lines.append(line_text.strip())
        except (IndexError, TypeError, KeyError):
            continue

    return lines, "\n".join(lines)


async def run_ocr_on_file(image_path: Path) -> tuple[list[str], str]:
    """Run OCR on a single image file.

    The asyncio lock guarantees sequential GPU access — no concurrent inference.
    """
    async with _ocr_lock:
        ocr = get_ocr()
        logger.debug("Running OCR on %s", image_path)
        try:
            if hasattr(ocr, "predict"):
                result = ocr.predict(input=str(image_path))
            else:
                result = ocr.ocr(str(image_path), cls=True)
        except Exception as exc:
            logger.exception("OCR inference failed on %s", image_path)
            raise RuntimeError(f"OCR internal error: {exc}") from exc

    return extract_lines_and_text(result)
