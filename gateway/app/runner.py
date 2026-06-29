import logging
import time
from pathlib import Path

import httpx

from app.backends import BackendManager
from app.config import settings
from app.inputs import render_to_images, resolve_input_path
from app.ocr_client import ocr_pages
from app.outputs import write_markdown
from app.registry import ModelSpec, get_model, model_names
from app.schemas import ModelResult

logger = logging.getLogger(__name__)


def _display_path(path: Path) -> str:
    """Render an output path relative to the mount root for the API response."""
    try:
        return str(path.relative_to(settings.output_dir.parent))
    except ValueError:
        return str(path)


async def run_ocr(
    backends: BackendManager,
    model: str,
    rel_path: str,
    keep_loaded: bool,
) -> list[ModelResult]:
    """Run OCR for one model or every model (``model == "all"``).

    The input is resolved once and rendered to page images once; each model then
    OCRs the same pages. For ``"all"``, models run sequentially, swapping the GPU
    backend between them. A failure on one model is captured in its result and
    does not abort the others.

    Args:
        backends: The shared backend lifecycle manager.
        model: A configured model name, or ``"all"``.
        rel_path: Input file path relative to the data dir.
        keep_loaded: Keep the last backend running after finishing.

    Returns:
        One :class:`ModelResult` per model that was run.
    """
    input_path = resolve_input_path(rel_path)
    images, temp_paths = render_to_images(input_path)

    targets = model_names() if model == "all" else [model]

    results: list[ModelResult] = []
    try:
        async with httpx.AsyncClient() as client:
            for name in targets:
                spec = get_model(name)
                results.append(
                    await _run_one(backends, client, spec, input_path, images)
                )
    finally:
        for tmp in temp_paths:
            tmp.unlink(missing_ok=True)
        if not keep_loaded and backends.active is not None:
            await backends.stop(backends.active)

    return results


async def _run_one(
    backends: BackendManager,
    client: httpx.AsyncClient,
    spec: ModelSpec,
    input_path: Path,
    images: list[Path],
) -> ModelResult:
    t0 = time.perf_counter()
    try:
        await backends.ensure_only(spec)
        markdown = await ocr_pages(client, spec, images)
        out_path = write_markdown(spec.name, input_path, markdown)
        return ModelResult(
            model=spec.name,
            status="ok",
            output_path=_display_path(out_path),
            chars=len(markdown),
            pages=len(images),
            elapsed_s=round(time.perf_counter() - t0, 2),
        )
    except Exception as exc:  # noqa: BLE001 - report per-model, keep others running
        logger.exception("Model '%s' failed on %s", spec.name, input_path.name)
        return ModelResult(
            model=spec.name,
            status="error",
            pages=len(images),
            elapsed_s=round(time.perf_counter() - t0, 2),
            error=str(exc),
        )
