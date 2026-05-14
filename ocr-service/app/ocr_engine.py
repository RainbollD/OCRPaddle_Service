import asyncio
import gc
import logging
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app.config import settings
from app.increasing_image_quality import improve_image_for_ocr
from app.schemas import ContentBlock, TableBlock, TableCell, TableRow
from paddleocr import PPStructureV3

logger = logging.getLogger(__name__)

_structured_ocr_instance: Any | None = None
_structured_ocr_device: str | None = None
_structured_ocr_lock = asyncio.Lock()


def _clear_device_cache() -> None:
    """Release Python references and ask Paddle to return cached GPU memory."""
    gc.collect()
    try:
        import paddle

        cuda_module = getattr(getattr(paddle, "device", None), "cuda", None)
        empty_cache = getattr(cuda_module, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        logger.debug("Failed to clear Paddle device cache", exc_info=True)


def _create_structured_ocr_instance(device: str) -> Any:
    """Instantiate PPStructureV3 with project-configured models."""

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", settings.paddle_pdx_model_source)

    logger.info(
        "Initialising PPStructureV3: det=%s rec=%s lang=%s device=%s",
        settings.text_detection_model,
        settings.text_recognition_model,
        settings.ocr_lang,
        device,
    )

    return PPStructureV3(
        text_detection_model_name=settings.text_detection_model,
        text_recognition_model_name=settings.text_recognition_model,
        lang=settings.ocr_lang,
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _ensure_structured_ocr_loaded(device: str | None = None) -> Any:
    """Load PPStructureV3 lazily and keep a single structured OCR instance."""
    global _structured_ocr_device, _structured_ocr_instance

    target_device = device or settings.ocr_device
    if (
        _structured_ocr_instance is None
        or _structured_ocr_device is None
        or _structured_ocr_device != target_device
    ):
        if _structured_ocr_instance is not None:
            logger.info(
                "Releasing PPStructureV3 before switching device: %s -> %s",
                _structured_ocr_device,
                target_device,
            )
            _structured_ocr_instance = None
            _structured_ocr_device = None
            _clear_device_cache()

        _structured_ocr_instance = _create_structured_ocr_instance(target_device)
        _structured_ocr_device = target_device
        logger.info("PPStructureV3 engine ready on %s.", target_device)

    return _structured_ocr_instance


def get_structured_ocr() -> Any:
    """Return the structured OCR singleton; raises RuntimeError if not yet initialised."""
    if _structured_ocr_instance is None:
        raise RuntimeError("Structured OCR engine has not been initialised yet.")
    return _structured_ocr_instance


def init_structured_ocr() -> None:
    """Called once at application startup to warm up the PPStructureV3 model."""
    _ensure_structured_ocr_loaded()


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


def _callable_or_value(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _block_as_mapping(block: Any) -> dict[str, Any] | None:
    """Normalise a PaddleX parsing block into a mapping.

    PaddleX may expose parsing blocks either as dictionaries, objects with
    `.json()`, or objects with public attributes. The parser works with one
    mapping shape so it can handle all of those variants.
    """
    mapping = _as_mapping(block)
    if mapping is not None:
        return mapping

    attrs: dict[str, Any] = {}
    for attr_name in (
        "block_type",
        "block_label",
        "type",
        "block_content",
        "content",
        "text",
        "html",
        "table_html",
        "pred_html",
    ):
        if not hasattr(block, attr_name):
            continue

        value = _callable_or_value(getattr(block, attr_name))
        if value is not None:
            attrs[attr_name] = value

    return attrs or None


def _first_mapping_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


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
        block_mapping = _block_as_mapping(block)
        if block_mapping is None:
            continue
        lines.extend(_lines_from_text(_first_mapping_value(
            block_mapping,
            ("block_content", "content", "text", "html", "table_html", "pred_html"),
        )))

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


# ---------------------------------------------------------------------------
# HTML table parser (stdlib — no extra deps)
# ---------------------------------------------------------------------------

class _TableHTMLParser(HTMLParser):
    """Minimal HTML parser that extracts rows and cells from a <table> element."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: str | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = ""
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_cell:
            cell_text = (self._current_cell or "").strip()
            if self._current_row is not None:
                self._current_row.append(cell_text)
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            if any(self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            self._current_cell += data


def _html_table_to_block(html: str) -> TableBlock:
    """Parse an HTML table string into a structured TableBlock."""
    parser = _TableHTMLParser()
    parser.feed(html)

    schema_rows: list[TableRow] = [
        TableRow(cells=[TableCell(content=cell) for cell in row])
        for row in parser.rows
    ]

    # Build a markdown table from the parsed rows.
    md_lines: list[str] = []
    for idx, row in enumerate(parser.rows):
        md_lines.append("| " + " | ".join(row) + " |")
        if idx == 0:
            md_lines.append("| " + " | ".join("---" for _ in row) + " |")

    return TableBlock(
        rows=schema_rows,
        markdown="\n".join(md_lines),
        html=html,
    )


def _block_type_label(raw_type: str) -> str:
    """Normalise PPStructureV3 block_type values to our schema literals."""
    t = raw_type.lower()
    if t == "table":
        return "table"
    if t in ("figure", "image", "chart"):
        return "figure"
    if t in ("formula", "equation"):
        return "formula"
    if t in ("text", "title", "paragraph", "header", "footer", "caption"):
        return "text"
    return "other"


def _extract_full_markdown(result_item: Any) -> str:
    """Pull the ready-made markdown string that PPStructureV3 builds."""
    markdown_attr = (
        result_item.get("markdown") if isinstance(result_item, dict)
        else getattr(result_item, "markdown", None)
    )
    if callable(markdown_attr):
        try:
            markdown_attr = markdown_attr()
        except TypeError:
            markdown_attr = None

    if isinstance(markdown_attr, dict):
        return (markdown_attr.get("markdown_texts") or markdown_attr.get("text") or "")
    if isinstance(markdown_attr, str):
        return markdown_attr
    return ""


def _debug_mapping_shape(item: Any) -> dict[str, Any]:
    """Return a compact, log-safe description of one PPStructureV3 result item."""
    mapping = _as_mapping(item)
    shape: dict[str, Any] = {
        "item_type": type(item).__name__,
        "mapping_found": mapping is not None,
    }

    if mapping is None:
        shape["public_attrs"] = [
            attr for attr in dir(item)
            if not attr.startswith("_")
        ][:20]
        return shape

    shape["mapping_keys"] = sorted(str(key) for key in mapping.keys())[:30]
    nested = mapping.get("res")
    if isinstance(nested, dict):
        shape["res_keys"] = sorted(str(key) for key in nested.keys())[:30]
        parsing_list = nested.get("parsing_res_list")
    else:
        parsing_list = mapping.get("parsing_res_list")

    if isinstance(parsing_list, list):
        shape["parsing_res_list_len"] = len(parsing_list)
        if parsing_list:
            first_block = parsing_list[0]
            first_block_mapping = _block_as_mapping(first_block)
            shape["first_block_item_type"] = type(first_block).__name__
        if parsing_list and first_block_mapping is not None:
            shape["first_block_keys"] = sorted(
                str(key) for key in first_block_mapping.keys()
            )[:30]
            shape["first_block_type"] = (
                first_block_mapping.get("block_type")
                or first_block_mapping.get("block_label")
                or first_block_mapping.get("type")
            )

    markdown = (
        item.get("markdown") if isinstance(item, dict)
        else getattr(item, "markdown", None)
    )
    shape["markdown_attr_type"] = type(markdown).__name__
    return shape


def _log_empty_structured_result(ocr_result: Any) -> None:
    """Log the raw PPStructureV3 result shape when parsing produced no content."""
    if not logger.isEnabledFor(logging.WARNING):
        return

    items = ocr_result if isinstance(ocr_result, list) else [ocr_result]
    logger.warning(
        "PPStructureV3 returned no parsed content. result_type=%s item_count=%d shapes=%s",
        type(ocr_result).__name__,
        len(items),
        [_debug_mapping_shape(item) for item in items[:3]],
    )


def extract_structured_content(ocr_result: Any) -> tuple[list[ContentBlock], str, str]:
    """Parse PPStructureV3 output into typed ContentBlock list, full text, and full markdown.

    Returns:
        blocks:        Ordered list of ContentBlock (text / table / figure / …).
        full_text:     All text content joined with newlines.
        full_markdown: Full markdown document string (tables as Markdown tables).
    """
    blocks: list[ContentBlock] = []

    if not ocr_result:
        logger.warning("PPStructureV3 returned an empty result object: %r", ocr_result)
        return blocks, "", ""

    items = ocr_result if isinstance(ocr_result, list) else [ocr_result]

    for item in items:
        # Try to get the full markdown from the engine's own renderer first.
        engine_markdown = _extract_full_markdown(item)

        mapping = _as_mapping(item)
        if mapping is None:
            continue

        nested = mapping.get("res")
        if isinstance(nested, dict):
            mapping = nested

        parsing_list = mapping.get("parsing_res_list")
        if not isinstance(parsing_list, list):
            # Fallback: treat the whole item as a single text block.
            text_lines = _extract_from_mapping(item)
            text = "\n".join(text_lines)
            if text:
                blocks.append(ContentBlock(type="text", text=text, table=None))
            full_text = "\n".join(b.text for b in blocks if b.text)
            if not blocks and not full_text and not engine_markdown:
                _log_empty_structured_result(ocr_result)
            return blocks, full_text, engine_markdown or full_text

        for raw_block in parsing_list:
            block_mapping = _block_as_mapping(raw_block)
            if block_mapping is None:
                continue

            raw_type = str(
                _first_mapping_value(
                    block_mapping,
                    ("block_type", "block_label", "type"),
                ) or "other"
            )
            block_label = _block_type_label(raw_type)
            block_content = _as_text(_first_mapping_value(
                block_mapping,
                ("block_content", "content", "text", "html", "table_html", "pred_html"),
            ))

            if block_label == "table":
                table_block = _html_table_to_block(block_content)
                # Plain-text version: rows joined by newlines, cells by spaces.
                plain_rows = [
                    "  ".join(cell.content for cell in row.cells)
                    for row in table_block.rows
                ]
                plain_text = "\n".join(plain_rows)
                blocks.append(ContentBlock(type="table", text=plain_text, table=table_block))
            else:
                text = "\n".join(_lines_from_text(block_content))
                if text or block_label == "figure":
                    blocks.append(ContentBlock(type=block_label, text=text, table=None))  # type: ignore[arg-type]

        full_text = "\n".join(b.text for b in blocks if b.text)

        # Prefer the engine-rendered markdown; fall back to assembling it from blocks.
        if not engine_markdown:
            md_parts: list[str] = []
            for b in blocks:
                if b.type == "table" and b.table:
                    md_parts.append(b.table.markdown)
                elif b.text:
                    md_parts.append(b.text)
            engine_markdown = "\n\n".join(md_parts)

        if not blocks and not full_text and not engine_markdown:
            _log_empty_structured_result(ocr_result)
        return blocks, full_text, engine_markdown

    full_text = "\n".join(b.text for b in blocks if b.text)
    if not blocks and not full_text:
        _log_empty_structured_result(ocr_result)
    return blocks, full_text, full_text


async def run_structured_ocr_on_file(
    image_path: Path,
    device: str | None = None,
) -> tuple[list[ContentBlock], str, str]:
    """Run layout-aware OCR (PPStructureV3) on a single image file.

    Returns:
        blocks:        Typed content blocks (text / table / figure / …).
        full_text:     All text content joined with newlines.
        full_markdown: Markdown document with tables rendered as Markdown tables.

    The asyncio lock guarantees sequential GPU access — no concurrent inference.
    """
    try:
        ocr_input_path = await improve_image_for_ocr(image_path)
    except Exception as exc:
        logger.exception("Image preprocessing failed on %s", image_path)
        raise RuntimeError(f"OCR preprocessing error: {exc}") from exc

    async with _structured_ocr_lock:
        ocr = _ensure_structured_ocr_loaded(device)
        logger.debug("Running structured OCR on %s with device=%s", ocr_input_path, device)
        try:
            result = ocr.predict(input=str(ocr_input_path))
        except Exception as exc:
            logger.exception("Structured OCR inference failed on %s", ocr_input_path)
            raise RuntimeError(f"Structured OCR internal error: {exc}") from exc

    return extract_structured_content(result)
