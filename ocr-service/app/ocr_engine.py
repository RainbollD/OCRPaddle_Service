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

logger = logging.getLogger(__name__)

_structured_ocr_instance: Any | None = None
_structured_ocr_device: str | None = None
_structured_ocr_lock = asyncio.Lock()


def _log_value_summary(value: Any, *, max_text_len: int = 300) -> Any:
    """Return a compact value description for readable OCR trace logs."""
    if value is None:
        return None
    if isinstance(value, str):
        return {
            "type": "str",
            "len": len(value),
            "sample": value[:max_text_len],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "len": len(value),
            "keys": sorted(str(key) for key in value.keys())[:30],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "len": len(value),
            "first_type": type(value[0]).__name__ if value else None,
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "len": len(value),
            "first_type": type(value[0]).__name__ if value else None,
        }
    return {
        "type": type(value).__name__,
        "repr": repr(value)[:max_text_len],
    }


def _clear_device_cache() -> None:
    """Release Python references and cached Paddle GPU memory.

    The function is best-effort: cleanup failures are logged at debug level and
    never propagated because cache cleanup should not break request handling.

    Returns:
        None: This function only performs cleanup side effects.
    """
    logger.info("Clearing OCR device cache.")
    collected = gc.collect()
    logger.info("Python garbage collector finished. collected_objects=%d", collected)
    try:
        import paddle

        cuda_module = getattr(getattr(paddle, "device", None), "cuda", None)
        empty_cache = getattr(cuda_module, "empty_cache", None)
        logger.info(
            "Paddle CUDA cache cleanup lookup. cuda_module=%s empty_cache_callable=%s",
            type(cuda_module).__name__ if cuda_module is not None else None,
            callable(empty_cache),
        )
        if callable(empty_cache):
            empty_cache()
            logger.info("Paddle CUDA empty_cache() finished.")
    except Exception:
        logger.debug("Failed to clear Paddle device cache", exc_info=True)


def _create_structured_ocr_instance(device: str) -> Any:
    """Create a configured PPStructureV3 OCR engine instance.

    PaddleOCR is imported lazily here so importing this module does not load GPU
    libraries or model code. That keeps health checks and parser tests cheap.

    Args:
        device (str): Paddle device string such as ``cpu`` or ``gpu:0``.

    Returns:
        Any: Initialized PPStructureV3 instance.
    """
    # Imported here so that merely importing ocr_engine does not load PaddleOCR.
    # This keeps the /health endpoint and unit tests free of GPU dependencies.
    logger.info("Importing PPStructureV3 for lazy OCR engine creation.")
    from paddleocr import PPStructureV3  # noqa: PLC0415

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", settings.paddle_pdx_model_source)
    logger.info(
        "Paddle model source prepared. PADDLE_PDX_MODEL_SOURCE=%s",
        os.environ.get("PADDLE_PDX_MODEL_SOURCE"),
    )

    logger.info(
        "Initialising PPStructureV3: det=%s rec=%s lang=%s device=%s",
        settings.text_detection_model,
        settings.text_recognition_model,
        settings.ocr_lang,
        device,
    )

    ocr_instance = PPStructureV3(
        text_detection_model_name=settings.text_detection_model,
        text_recognition_model_name=settings.text_recognition_model,
        lang=settings.ocr_lang,
        device=device,

        # Orientation handling.
        use_doc_orientation_classify=True,
        use_textline_orientation=True,
        use_doc_unwarping=False,

        # Structure modules.
        use_table_recognition=True,
        use_formula_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,

        # Detection tuning for low-quality scans.
        text_det_limit_side_len=1280,
        text_det_limit_type="max",
    )
    logger.info("PPStructureV3 instance created. instance_type=%s", type(ocr_instance).__name__)
    return ocr_instance


def _ensure_structured_ocr_loaded(device: str | None = None) -> Any:
    """Load PPStructureV3 lazily and reuse one instance per device.

    If the requested device changes, the previous engine is released before a
    new one is created. This avoids holding GPU memory for stale instances.

    Args:
        device (str | None, optional): Requested OCR device. Defaults to the
            configured ``settings.ocr_device``.

    Returns:
        Any: Loaded PPStructureV3 engine instance.
    """
    global _structured_ocr_device, _structured_ocr_instance

    target_device = device or settings.ocr_device
    logger.info(
        "Ensuring structured OCR is loaded. requested_device=%s target_device=%s current_device=%s has_instance=%s",
        device,
        target_device,
        _structured_ocr_device,
        _structured_ocr_instance is not None,
    )
    if (
        _structured_ocr_instance is None
        or _structured_ocr_device is None
        or _structured_ocr_device != target_device
    ):
        logger.info("Structured OCR instance needs initialization or device switch.")
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
    else:
        logger.info("Reusing existing PPStructureV3 instance on %s.", _structured_ocr_device)

    return _structured_ocr_instance


def get_structured_ocr() -> Any:
    """Return the already initialized structured OCR singleton.

    Returns:
        Any: Current PPStructureV3 engine instance.

    Raises:
        RuntimeError: If the OCR engine has not been initialized yet.
    """
    logger.info(
        "Returning structured OCR singleton. has_instance=%s device=%s",
        _structured_ocr_instance is not None,
        _structured_ocr_device,
    )
    if _structured_ocr_instance is None:
        raise RuntimeError("Structured OCR engine has not been initialised yet.")
    return _structured_ocr_instance


def _as_mapping(item: Any) -> dict[str, Any] | None:
    """Return a dictionary view of a PPStructureV3 result item when possible.

    Args:
        item (Any): Raw OCR result item, dictionary, or Paddle result object.

    Returns:
        dict[str, Any] | None: Mapping representation of the item, or ``None``
            when the shape is unsupported.
    """
    logger.info("Trying to convert OCR item to mapping. item=%s", _log_value_summary(item))
    if isinstance(item, dict):
        logger.info("OCR item is already a dict. keys=%s", sorted(str(key) for key in item.keys())[:30])
        return item

    json_data = getattr(item, "json", None)
    logger.info(
        "OCR item json attribute inspected. item_type=%s json_attr=%s",
        type(item).__name__,
        _log_value_summary(json_data),
    )
    if callable(json_data):
        try:
            json_data = json_data()
            logger.info("OCR item json() called successfully. result=%s", _log_value_summary(json_data))
        except TypeError:
            logger.info("OCR item json attribute is callable but requires arguments.")
            json_data = None

    if isinstance(json_data, dict):
        nested = json_data.get("res")
        logger.info(
            "OCR item json data is dict. nested_res_is_dict=%s json_keys=%s",
            isinstance(nested, dict),
            sorted(str(key) for key in json_data.keys())[:30],
        )
        return nested if isinstance(nested, dict) else json_data

    logger.info("OCR item could not be converted to mapping. json_data=%s", _log_value_summary(json_data))
    return None


def _callable_or_value(value: Any) -> Any:
    """Evaluate no-argument callables and return plain values unchanged.

    Args:
        value (Any): Attribute value or callable attribute from a Paddle object.

    Returns:
        Any: Callable result, original value, or ``None`` when the callable
            requires arguments.
    """
    if callable(value):
        try:
            result = value()
            logger.info(
                "Callable OCR attribute evaluated. callable_type=%s result=%s",
                type(value).__name__,
                _log_value_summary(result),
            )
            return result
        except TypeError:
            logger.info("Callable OCR attribute skipped because it requires arguments.")
            return None
    logger.info("OCR attribute used as plain value. value=%s", _log_value_summary(value))
    return value


def _block_as_mapping(block: Any) -> dict[str, Any] | None:
    """Normalise a PaddleX parsing block into a mapping.

    PaddleX may expose parsing blocks either as dictionaries, objects with
    `.json()`, or objects with public attributes. The parser works with one
    mapping shape so it can handle all of those variants.

    Args:
        block (Any): Raw parsing block returned by PaddleX.

    Returns:
        dict[str, Any] | None: Normalized block mapping, or ``None`` when no
            supported fields can be extracted.
    """
    logger.info("Normalising parsing block. block=%s", _log_value_summary(block))
    mapping = _as_mapping(block)
    if mapping is not None:
        logger.info(
            "Parsing block normalized from mapping/json. keys=%s",
            sorted(str(key) for key in mapping.keys())[:30],
        )
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
            logger.info(
                "Parsing block attribute captured. attr_name=%s value=%s",
                attr_name,
                _log_value_summary(value),
            )

    logger.info("Parsing block normalized from attributes. attrs=%s", _log_value_summary(attrs))
    return attrs or None


def _first_mapping_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty mapping value found for the given keys.

    Args:
        mapping (dict[str, Any]): Mapping to inspect.
        keys (tuple[str, ...]): Candidate keys in priority order.

    Returns:
        Any: First value whose key exists and is not ``None``, or ``None`` when
            none of the keys are present.
    """
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            logger.info(
                "First mapping value found. key=%s value=%s",
                key,
                _log_value_summary(value),
            )
            return value
    logger.info("No mapping value found for keys=%s. mapping_keys=%s", keys, sorted(str(key) for key in mapping.keys())[:30])
    return None


def _as_text(value: Any) -> str:
    """Convert optional OCR values to plain text.

    Args:
        value (Any): OCR value that may be ``None``, text, or another object.

    Returns:
        str: Empty string for ``None`` values, otherwise a string
            representation of the value.
    """
    if value is None:
        logger.info("Converting OCR value to text. value=None result_len=0")
        return ""
    if isinstance(value, str):
        logger.info("Converting OCR value to text. value_type=str result_len=%d", len(value))
        return value
    text = str(value)
    logger.info(
        "Converting OCR value to text. value_type=%s result_len=%d sample=%r",
        type(value).__name__,
        len(text),
        text[:300],
    )
    return text


def _lines_from_text(text: Any) -> list[str]:
    """Split a text value into stripped non-empty lines.

    Args:
        text (Any): Candidate text value to split.

    Returns:
        list[str]: Non-empty stripped lines, or an empty list for non-string
            values.
    """
    if not isinstance(text, str):
        logger.info(
            "Splitting text into lines skipped. input_type=%s",
            type(text).__name__,
        )
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    logger.info(
        "Split text into non-empty lines. input_len=%d line_count=%d first_line=%r",
        len(text),
        len(lines),
        lines[0][:300] if lines else None,
    )
    return lines


def _extract_markdown_lines(item: Any) -> list[str]:
    """Extract markdown-rendered text lines from a PPStructureV3 result item.

    Args:
        item (Any): Raw OCR result item with an optional ``markdown`` payload.

    Returns:
        list[str]: Non-empty markdown text lines extracted from the result.
    """
    markdown = (
        item.get("markdown") if isinstance(item, dict) else getattr(item, "markdown", None)
    )
    logger.info("Extracting markdown lines. markdown_attr=%s", _log_value_summary(markdown))
    if callable(markdown):
        try:
            markdown = markdown()
            logger.info("Markdown callable evaluated. markdown=%s", _log_value_summary(markdown))
        except TypeError:
            logger.info("Markdown callable skipped because it requires arguments.")
            markdown = None

    if not isinstance(markdown, dict):
        logger.info(
            "Markdown lines unavailable because markdown is not dict. markdown=%s",
            _log_value_summary(markdown),
        )
        return []

    markdown_value = markdown.get("markdown_texts") or markdown.get("text")
    lines = _lines_from_text(markdown_value)
    logger.info("Markdown lines extracted. line_count=%d", len(lines))
    return lines


def _extract_rec_texts(mapping: dict[str, Any]) -> list[str]:
    """Extract recognized text lines from an OCR result mapping.

    Args:
        mapping (dict[str, Any]): OCR result mapping that may contain
            ``rec_texts``.

    Returns:
        list[str]: Stripped recognized text lines.
    """
    rec_texts = mapping.get("rec_texts")
    logger.info(
        "Extracting rec_texts. rec_texts=%s mapping_keys=%s",
        _log_value_summary(rec_texts),
        sorted(str(key) for key in mapping.keys())[:30],
    )
    if isinstance(rec_texts, list):
        lines = [str(text).strip() for text in rec_texts if str(text).strip()]
        logger.info("rec_texts extracted. line_count=%d first_line=%r", len(lines), lines[0][:300] if lines else None)
        return lines

    logger.info("rec_texts skipped because value is not a list.")
    return []


def _extract_parsing_blocks(mapping: dict[str, Any]) -> list[str]:
    """Extract text lines from PaddleX parsing blocks.

    Args:
        mapping (dict[str, Any]): OCR result mapping that may contain
            ``parsing_res_list``.

    Returns:
        list[str]: Text lines extracted from all supported parsing blocks.
    """
    blocks = mapping.get("parsing_res_list")
    logger.info(
        "Extracting parsing blocks. parsing_res_list=%s",
        _log_value_summary(blocks),
    )
    if not isinstance(blocks, list):
        logger.info("Parsing blocks skipped because parsing_res_list is not a list.")
        return []

    lines: list[str] = []
    for block_index, block in enumerate(blocks):
        logger.info("Parsing fallback block started. block_index=%d block=%s", block_index, _log_value_summary(block))
        block_mapping = _block_as_mapping(block)
        if block_mapping is None:
            logger.info("Parsing fallback block skipped. block_index=%d reason=no_mapping", block_index)
            continue
        block_value = _first_mapping_value(
            block_mapping,
            ("block_content", "content", "text", "html", "table_html", "pred_html"),
        )
        block_lines = _lines_from_text(block_value)
        lines.extend(block_lines)
        logger.info(
            "Parsing fallback block finished. block_index=%d block_line_count=%d total_line_count=%d",
            block_index,
            len(block_lines),
            len(lines),
        )

    logger.info("Parsing blocks extracted. total_line_count=%d", len(lines))
    return lines


def _extract_from_mapping(item: Any) -> list[str]:
    """Extract fallback text lines from any supported result mapping shape.

    The parser tries engine markdown first, then structured parsing blocks, then
    known nested OCR result containers, and finally simple text fields.

    Args:
        item (Any): Raw OCR result item.

    Returns:
        list[str]: Best-effort extracted text lines.
    """
    logger.info("Extracting fallback text from OCR item. item=%s", _log_value_summary(item))
    markdown_lines = _extract_markdown_lines(item)
    if markdown_lines:
        logger.info("Fallback text source selected: markdown. line_count=%d", len(markdown_lines))
        return markdown_lines

    mapping = _as_mapping(item)
    if mapping is None:
        logger.info("Fallback text extraction stopped: item has no mapping.")
        return []

    nested = mapping.get("res")
    if isinstance(nested, dict):
        logger.info("Fallback text extraction using nested res mapping. res_keys=%s", sorted(str(key) for key in nested.keys())[:30])
        mapping = nested

    parsing_lines = _extract_parsing_blocks(mapping)
    if parsing_lines:
        logger.info("Fallback text source selected: parsing_res_list. line_count=%d", len(parsing_lines))
        return parsing_lines

    for nested_key in ("overall_ocr_res", "text_paragraphs_ocr_res"):
        nested_result = mapping.get(nested_key)
        logger.info(
            "Checking nested OCR fallback. nested_key=%s nested_result=%s",
            nested_key,
            _log_value_summary(nested_result),
        )
        if isinstance(nested_result, dict):
            rec_lines = _extract_rec_texts(nested_result)
            if rec_lines:
                logger.info(
                    "Fallback text source selected: %s.rec_texts. line_count=%d",
                    nested_key,
                    len(rec_lines),
                )
                return rec_lines

    rec_lines = _extract_rec_texts(mapping)
    if rec_lines:
        logger.info("Fallback text source selected: root rec_texts. line_count=%d", len(rec_lines))
        return rec_lines

    line_text = mapping.get("transcription") or mapping.get("text")
    if line_text and str(line_text).strip():
        lines = [str(line_text).strip()]
        logger.info(
            "Fallback text source selected: transcription/text. line_count=%d first_line=%r",
            len(lines),
            lines[0][:300],
        )
        return lines

    logger.info("Fallback text extraction found no usable text.")
    return []


# ---------------------------------------------------------------------------
# HTML table parser (stdlib — no extra deps)
# ---------------------------------------------------------------------------

class _TableHTMLParser(HTMLParser):
    """Minimal HTML parser that extracts rows and cells from a <table> element."""

    def __init__(self) -> None:
        """Initialize parser state for accumulating table rows and cells.

        Returns:
            None: The parser stores extracted rows on ``self.rows``.
        """
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: str | None = None
        self._in_cell = False
        logger.info("HTML table parser initialized.")

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Start collecting row or cell content for table tags.

        Args:
            tag (str): HTML tag name encountered by ``HTMLParser``.
            attrs (list): Raw HTML attributes for the tag.

        Returns:
            None: Parser state is updated in place.
        """
        if tag == "tr":
            self._current_row = []
            logger.info("HTML table parser started row.")
        elif tag in ("td", "th"):
            self._current_cell = ""
            self._in_cell = True
            logger.info("HTML table parser started cell. tag=%s", tag)

    def handle_endtag(self, tag: str) -> None:
        """Finalize row or cell content when a table tag closes.

        Args:
            tag (str): HTML tag name whose closing token was encountered.

        Returns:
            None: Completed rows are appended to ``self.rows``.
        """
        if tag in ("td", "th") and self._in_cell:
            cell_text = (self._current_cell or "").strip()
            if self._current_row is not None:
                self._current_row.append(cell_text)
                logger.info(
                    "HTML table parser finished cell. tag=%s cell_len=%d current_row_cells=%d sample=%r",
                    tag,
                    len(cell_text),
                    len(self._current_row),
                    cell_text[:300],
                )
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            if any(self._current_row):
                self.rows.append(self._current_row)
                logger.info(
                    "HTML table parser finished row. row_cells=%d total_rows=%d",
                    len(self._current_row),
                    len(self.rows),
                )
            self._current_row = None

    def handle_data(self, data: str) -> None:
        """Append text data to the current table cell.

        Args:
            data (str): Text content emitted by ``HTMLParser``.

        Returns:
            None: Current cell text is updated in place.
        """
        if self._in_cell and self._current_cell is not None:
            self._current_cell += data
            logger.info(
                "HTML table parser received cell text. chunk_len=%d current_cell_len=%d sample=%r",
                len(data),
                len(self._current_cell),
                data[:300],
            )


def _html_table_to_block(html: str) -> TableBlock:
    """Parse an HTML table string into a structured table block.

    Args:
        html (str): HTML table markup returned by PPStructureV3.

    Returns:
        TableBlock: Parsed rows, generated markdown, and original HTML.
    """
    logger.info("Parsing HTML table into block. html_len=%d sample=%r", len(html), html[:300])
    parser = _TableHTMLParser()
    parser.feed(html)
    logger.info("HTML table parsed. row_count=%d", len(parser.rows))

    schema_rows: list[TableRow] = [
        TableRow(cells=[TableCell(content=cell) for cell in row])
        for row in parser.rows
    ]
    logger.info("Table schema rows built. row_count=%d", len(schema_rows))

    # Build a markdown table from the parsed rows.
    md_lines: list[str] = []
    for idx, row in enumerate(parser.rows):
        md_lines.append("| " + " | ".join(row) + " |")
        if idx == 0:
            md_lines.append("| " + " | ".join("---" for _ in row) + " |")
        logger.info(
            "Table markdown row built. row_index=%d cell_count=%d md_line_count=%d",
            idx,
            len(row),
            len(md_lines),
        )

    table_block = TableBlock(
        rows=schema_rows,
        markdown="\n".join(md_lines),
        html=html,
    )
    logger.info(
        "Table block created. rows=%d markdown_len=%d html_len=%d",
        len(table_block.rows),
        len(table_block.markdown),
        len(table_block.html),
    )
    return table_block


def _block_type_label(raw_type: str) -> str:
    """Normalize PPStructureV3 block type values to schema literals.

    Args:
        raw_type (str): Raw block type or label returned by PaddleX.

    Returns:
        str: One of ``text``, ``table``, ``figure``, ``formula``, or ``other``.
    """
    t = raw_type.lower()
    logger.info("Normalizing block type. raw_type=%r normalized_input=%r", raw_type, t)
    if t == "table":
        logger.info("Block type normalized. result=table")
        return "table"
    if t in ("figure", "image", "chart"):
        logger.info("Block type normalized. result=figure")
        return "figure"
    if t in ("formula", "equation"):
        logger.info("Block type normalized. result=formula")
        return "formula"
    if t in ("text", "title", "paragraph", "header", "footer", "caption"):
        logger.info("Block type normalized. result=text")
        return "text"
    logger.info("Block type normalized. result=other")
    return "other"


def _extract_full_markdown(result_item: Any) -> str:
    """Pull the ready-made markdown string that PPStructureV3 builds.

    Args:
        result_item (Any): Raw OCR result item with an optional markdown field.

    Returns:
        str: Engine-rendered markdown text, or an empty string when unavailable.
    """
    markdown_attr = (
        result_item.get("markdown") if isinstance(result_item, dict)
        else getattr(result_item, "markdown", None)
    )
    logger.info("Extracting full markdown. markdown_attr=%s", _log_value_summary(markdown_attr))
    if callable(markdown_attr):
        try:
            markdown_attr = markdown_attr()
            logger.info("Full markdown callable evaluated. markdown_attr=%s", _log_value_summary(markdown_attr))
        except TypeError:
            logger.info("Full markdown callable skipped because it requires arguments.")
            markdown_attr = None

    if isinstance(markdown_attr, dict):
        markdown_text = markdown_attr.get("markdown_texts") or markdown_attr.get("text") or ""
        logger.info(
            "Full markdown extracted from dict. markdown_len=%d source_keys=%s",
            len(markdown_text),
            sorted(str(key) for key in markdown_attr.keys())[:30],
        )
        return markdown_text
    if isinstance(markdown_attr, str):
        logger.info("Full markdown extracted from string. markdown_len=%d", len(markdown_attr))
        return markdown_attr
    logger.info("Full markdown unavailable. markdown_attr=%s", _log_value_summary(markdown_attr))
    return ""


def _debug_mapping_shape(item: Any) -> dict[str, Any]:
    """Return a compact, log-safe description of one OCR result item.

    Args:
        item (Any): Raw result item to inspect.

    Returns:
        dict[str, Any]: Summary of public attributes, mapping keys, and nested
            parsing metadata useful for warning logs.
    """
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
    """Log OCR result shape details when parsing produced no content.

    Args:
        ocr_result (Any): Raw PPStructureV3 result that could not be parsed into
            text or blocks.

    Returns:
        None: Diagnostic details are written to the logger when warnings are
            enabled.
    """
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

    The parser handles both structured ``parsing_res_list`` results and simpler
    OCR-only fallback shapes. Engine-rendered markdown is preferred when
    present because it usually preserves table formatting best.

    Args:
        ocr_result (Any): Raw result returned by ``PPStructureV3.predict``.

    Returns:
        tuple[list[ContentBlock], str, str]: Ordered content blocks, combined
            plain text, and combined markdown.
    """
    blocks: list[ContentBlock] = []
    markdown_parts: list[str] = []
    logger.info("Structured OCR parsing started. ocr_result=%s", _log_value_summary(ocr_result))

    if not ocr_result:
        logger.warning("PPStructureV3 returned an empty result object: %r", ocr_result)
        return blocks, "", ""

    items = ocr_result if isinstance(ocr_result, list) else [ocr_result]
    logger.info("Structured OCR result normalized to items. item_count=%d", len(items))

    for item_index, item in enumerate(items):
        # Try to get the full markdown from the engine's own renderer first.
        logger.info(
            "Structured OCR item parsing started. item_index=%d item=%s",
            item_index,
            _log_value_summary(item),
        )
        engine_markdown = _extract_full_markdown(item)

        logger.info(
            "Structured OCR item engine markdown extracted. item_index=%d markdown_len=%d",
            item_index,
            len(engine_markdown),
        )
        mapping = _as_mapping(item)
        logger.info(
            "Structured OCR item mapping extracted. item_index=%d mapping=%s",
            item_index,
            _log_value_summary(mapping),
        )
        if mapping is None:
            logger.info("Structured OCR item skipped. item_index=%d reason=no_mapping", item_index)
            continue

        nested = mapping.get("res")
        logger.info(
            "Structured OCR item nested res inspected. item_index=%d nested=%s",
            item_index,
            _log_value_summary(nested),
        )
        if isinstance(nested, dict):
            mapping = nested
            logger.info(
                "Structured OCR item using nested res mapping. item_index=%d res_keys=%s",
                item_index,
                sorted(str(key) for key in mapping.keys())[:30],
            )

        parsing_list = mapping.get("parsing_res_list")
        logger.info(
            "Structured OCR item parsing list inspected. item_index=%d parsing_list=%s",
            item_index,
            _log_value_summary(parsing_list),
        )
        if not isinstance(parsing_list, list):
            # Fallback: treat the whole item as a single text block.
            logger.info("Structured OCR item has no parsing list; using fallback extraction. item_index=%d", item_index)
            text_lines = _extract_from_mapping(item)
            logger.info(
                "Structured OCR fallback text lines extracted. item_index=%d line_count=%d",
                item_index,
                len(text_lines),
            )
            text = "\n".join(text_lines)
            logger.info(
                "Structured OCR fallback text joined. item_index=%d text_len=%d sample=%r",
                item_index,
                len(text),
                text[:300],
            )
            if text:
                blocks.append(ContentBlock(type="text", text=text, table=None))
                logger.info(
                    "Structured OCR fallback text block appended. item_index=%d total_blocks=%d",
                    item_index,
                    len(blocks),
                )
            if engine_markdown:
                markdown_parts.append(engine_markdown)
                logger.info(
                    "Structured OCR fallback markdown appended from engine. item_index=%d markdown_parts=%d",
                    item_index,
                    len(markdown_parts),
                )
            elif text:
                markdown_parts.append(text)
                logger.info(
                    "Structured OCR fallback markdown appended from text. item_index=%d markdown_parts=%d",
                    item_index,
                    len(markdown_parts),
                )
            continue

        item_block_start = len(blocks)
        logger.info(
            "Structured OCR parsing list processing started. item_index=%d block_start=%d block_count=%d",
            item_index,
            item_block_start,
            len(parsing_list),
        )
        for block_index, raw_block in enumerate(parsing_list):
            logger.info(
                "Structured OCR raw block started. item_index=%d block_index=%d raw_block=%s",
                item_index,
                block_index,
                _log_value_summary(raw_block),
            )
            block_mapping = _block_as_mapping(raw_block)
            logger.info(
                "Structured OCR block mapping extracted. item_index=%d block_index=%d block_mapping=%s",
                item_index,
                block_index,
                _log_value_summary(block_mapping),
            )
            if block_mapping is None:
                logger.info(
                    "Structured OCR raw block skipped. item_index=%d block_index=%d reason=no_mapping",
                    item_index,
                    block_index,
                )
                continue

            raw_type = str(
                _first_mapping_value(
                    block_mapping,
                    ("block_type", "block_label", "type"),
                ) or "other"
            )
            block_label = _block_type_label(raw_type)
            logger.info(
                "Structured OCR block type resolved. item_index=%d block_index=%d raw_type=%r block_label=%s",
                item_index,
                block_index,
                raw_type,
                block_label,
            )
            block_content = _as_text(_first_mapping_value(
                block_mapping,
                ("block_content", "content", "text", "html", "table_html", "pred_html"),
            ))
            logger.info(
                "Structured OCR block content extracted. item_index=%d block_index=%d content_len=%d sample=%r",
                item_index,
                block_index,
                len(block_content),
                block_content[:300],
            )

            if block_label == "table":
                table_block = _html_table_to_block(block_content)
                # Plain-text version: rows joined by newlines, cells by spaces.
                plain_rows = [
                    "  ".join(cell.content for cell in row.cells)
                    for row in table_block.rows
                ]
                plain_text = "\n".join(plain_rows)
                blocks.append(ContentBlock(type="table", text=plain_text, table=table_block))
                logger.info(
                    "Structured OCR table block appended. item_index=%d block_index=%d row_count=%d plain_text_len=%d total_blocks=%d",
                    item_index,
                    block_index,
                    len(table_block.rows),
                    len(plain_text),
                    len(blocks),
                )
            else:
                text = "\n".join(_lines_from_text(block_content))
                if text or block_label == "figure":
                    blocks.append(ContentBlock(type=block_label, text=text, table=None))  # type: ignore[arg-type]
                    logger.info(
                        "Structured OCR content block appended. item_index=%d block_index=%d type=%s text_len=%d total_blocks=%d",
                        item_index,
                        block_index,
                        block_label,
                        len(text),
                        len(blocks),
                    )
                else:
                    logger.info(
                        "Structured OCR content block skipped. item_index=%d block_index=%d type=%s reason=empty_text",
                        item_index,
                        block_index,
                        block_label,
                    )

        # Prefer the engine-rendered markdown; fall back to assembling it from blocks.
        if engine_markdown:
            markdown_parts.append(engine_markdown)
            logger.info(
                "Structured OCR item markdown appended from engine. item_index=%d markdown_len=%d markdown_parts=%d",
                item_index,
                len(engine_markdown),
                len(markdown_parts),
            )
        else:
            item_md: list[str] = []
            for b in blocks[item_block_start:]:
                if b.type == "table" and b.table:
                    item_md.append(b.table.markdown)
                elif b.text:
                    item_md.append(b.text)
            if item_md:
                markdown_parts.append("\n\n".join(item_md))
                logger.info(
                    "Structured OCR item markdown assembled from blocks. item_index=%d item_md_parts=%d markdown_parts=%d",
                    item_index,
                    len(item_md),
                    len(markdown_parts),
                )
            else:
                logger.info("Structured OCR item produced no markdown parts. item_index=%d", item_index)

    full_text = "\n".join(b.text for b in blocks if b.text)
    full_markdown = "\n\n".join(markdown_parts) if markdown_parts else full_text
    logger.info(
        "Structured OCR parsing finished. blocks=%d full_text_len=%d full_markdown_len=%d markdown_parts=%d",
        len(blocks),
        len(full_text),
        len(full_markdown),
        len(markdown_parts),
    )

    if not blocks and not full_text and not full_markdown:
        _log_empty_structured_result(ocr_result)
    return blocks, full_text, full_markdown


async def run_structured_ocr_on_file(
    image_path: Path,
    device: str | None = None,
) -> tuple[list[ContentBlock], str, str]:
    """Run layout-aware OCR (PPStructureV3) on a single image file.

    The image is preprocessed first, then inference runs under a single
    ``asyncio.Lock`` so GPU access remains sequential. The blocking PaddleOCR
    call is executed in a worker thread to keep the event loop responsive.

    Args:
        image_path (Path): Path to the image file to process.
        device (str | None, optional): Optional OCR device override such as
            ``cpu`` or ``gpu:0``. Defaults to ``None``.

    Returns:
        tuple[list[ContentBlock], str, str]: Parsed content blocks, combined
            plain text, and combined markdown.

    Raises:
        RuntimeError: If image preprocessing fails.
        RuntimeError: If structured OCR inference fails.
    """
    logger.info(
        "Structured OCR request started. image_path=%s device=%s",
        image_path,
        device,
    )
    try:
        ocr_input_path = await improve_image_for_ocr(image_path)
        logger.info(
            "Image preprocessing finished. original_path=%s ocr_input_path=%s",
            image_path,
            ocr_input_path,
        )
    except Exception as exc:
        logger.exception("Image preprocessing failed on %s", image_path)
        raise RuntimeError(f"OCR preprocessing error: {exc}") from exc

    logger.info("Waiting for structured OCR lock. ocr_input_path=%s", ocr_input_path)
    async with _structured_ocr_lock:
        logger.info("Structured OCR lock acquired. ocr_input_path=%s", ocr_input_path)
        ocr = _ensure_structured_ocr_loaded(device)
        logger.info(
            "Running structured OCR prediction. ocr_input_path=%s device=%s ocr_type=%s",
            ocr_input_path,
            device,
            type(ocr).__name__,
        )
        try:
            result = await asyncio.to_thread(
                lambda: ocr.predict(input=str(ocr_input_path))
            )
            logger.info(
                "Structured OCR prediction finished. result=%s",
                _log_value_summary(result),
            )
        except Exception as exc:
            logger.exception("Structured OCR inference failed on %s", ocr_input_path)
            raise RuntimeError(f"Structured OCR internal error: {exc}") from exc
        finally:
            logger.info("Structured OCR lock block finished. ocr_input_path=%s", ocr_input_path)

    parsed_result = extract_structured_content(result)
    blocks, full_text, full_markdown = parsed_result
    logger.info(
        "Structured OCR request finished. image_path=%s blocks=%d full_text_len=%d full_markdown_len=%d",
        image_path,
        len(blocks),
        len(full_text),
        len(full_markdown),
    )
    return parsed_result
