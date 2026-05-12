from pydantic import BaseModel


class ImageOCRResponse(BaseModel):
    filename: str
    text: str
    lines: list[str]
    engine: str = "paddleocr"
    language: str = "ru"


class PageResult(BaseModel):
    page: int
    text: str
    lines: list[str]


class PDFOCRResponse(BaseModel):
    filename: str
    pages: list[PageResult]
    full_text: str


class BatchImageResult(BaseModel):
    filename: str
    text: str
    lines: list[str]
    engine: str = "paddleocr"
    language: str = "ru"
    error: str | None = None


class BatchOCRResponse(BaseModel):
    results: list[BatchImageResult]


class HealthResponse(BaseModel):
    status: str
