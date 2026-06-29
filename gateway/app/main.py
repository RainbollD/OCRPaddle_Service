import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.backends import BackendError, BackendManager
from app.inputs import InputError
from app.registry import MODELS
from app.runner import run_ocr
from app.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    OCRRequest,
    OCRResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the backend manager on startup; backends load lazily per request."""
    logger.info("Starting OCR gateway. Backends start on demand.")
    app.state.backends = BackendManager()
    yield
    logger.info("Shutting down OCR gateway.")


app = FastAPI(
    title="OCR Comparison Gateway",
    description="Run OCR models by name on server-side files; output Markdown.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    """Lightweight readiness check that does not touch the GPU."""
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsResponse, tags=["Utility"])
async def list_models() -> ModelsResponse:
    """List configured models and which one is currently loaded."""
    active = app.state.backends.active
    return ModelsResponse(
        models=[
            ModelInfo(name=s.name, model_id=s.model_id, active=(s.name == active))
            for s in MODELS.values()
        ],
        active=active,
    )


@app.post("/ocr", response_model=OCRResponse, tags=["OCR"])
async def ocr(req: OCRRequest) -> OCRResponse:
    """Run OCR for one model (or ``all``) on a server-side file.

    Args:
        req: The OCR request (model name, file path, keep_loaded flag).

    Returns:
        Per-model results, including the path of each written Markdown file.

    Raises:
        HTTPException: 404/422 for bad input paths, 422 for unknown model names,
            503 when a backend cannot be started.
    """
    if req.model != "all" and req.model not in MODELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown model '{req.model}'. Known: {', '.join(MODELS)}, all.",
        )

    try:
        results = await run_ocr(
            app.state.backends, req.model, req.path, req.keep_loaded
        )
    except InputError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except BackendError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return OCRResponse(input=req.path, results=results)
