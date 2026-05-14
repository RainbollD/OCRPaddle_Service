from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Structured OCR schemas (table-aware)
# ---------------------------------------------------------------------------

class TableCell(BaseModel):
    content: str


class TableRow(BaseModel):
    cells: list[TableCell]


class TableBlock(BaseModel):
    rows: list[TableRow]
    markdown: str
    html: str


class ContentBlock(BaseModel):
    type: Literal["text", "table", "figure", "formula", "other"]
    text: str
    table: TableBlock | None = None


class StructuredImageOCRResponse(BaseModel):
    filename: str
    blocks: list[ContentBlock]
    text: str
    markdown: str
    language: str = "ru"
    engine: str = "ppstructurev3"


class StructuredPageResult(BaseModel):
    page: int
    blocks: list[ContentBlock]
    text: str
    markdown: str


class StructuredPDFOCRResponse(BaseModel):
    filename: str
    pages: list[StructuredPageResult]
    full_text: str
    full_markdown: str
