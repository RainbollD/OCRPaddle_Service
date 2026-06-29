from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ModelInfo(BaseModel):
    name: str
    model_id: str
    active: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    active: str | None = None


class OCRRequest(BaseModel):
    model: str = Field(
        ...,
        description="Model name (see /models) or 'all' to run every model.",
        examples=["deepseek-ocr", "all"],
    )
    path: str = Field(
        ...,
        description="Path to the input file, relative to the server's data dir.",
        examples=["scans/doc1.png"],
    )
    keep_loaded: bool = Field(
        default=True,
        description="Keep the backend running after the request (faster reuse).",
    )


class ModelResult(BaseModel):
    model: str
    status: Literal["ok", "error"]
    output_path: str | None = None
    chars: int | None = None
    pages: int | None = None
    elapsed_s: float | None = None
    error: str | None = None


class OCRResponse(BaseModel):
    input: str
    results: list[ModelResult]
