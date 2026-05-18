import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application startup and shutdown logging.

    The OCR models are intentionally not loaded here. They are initialized on
    the first OCR request so the health endpoint stays lightweight.

    Args:
        app (FastAPI): The FastAPI application instance managed by the
            lifespan context.

    Yields:
        None: Control is yielded back to FastAPI while the application runs.
    """
    logger.info("Starting up OCR service. OCR engines will be loaded on demand.")
    yield
    logger.info("Shutting down OCR service.")
    


app = FastAPI(
    title="OCR Service",
    description="Local structured OCR API backed by PPStructureV3.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
