import base64
import logging
import mimetypes
from pathlib import Path

import httpx

from app.config import settings
from app.registry import ModelSpec

logger = logging.getLogger(__name__)


def _image_data_url(path: Path) -> str:
    """Read an image and encode it as a base64 ``data:`` URL.

    Args:
        path: Path to the image file.

    Returns:
        A ``data:<mime>;base64,<...>`` URL suitable for the OpenAI
        ``image_url`` message content.
    """
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def ocr_image(client: httpx.AsyncClient, spec: ModelSpec, image: Path) -> str:
    """Run OCR on a single image via a backend's OpenAI-compatible API.

    Args:
        client: Shared async HTTP client.
        spec: The backend model specification.
        image: Path to the image to OCR.

    Returns:
        The Markdown text produced by the model.

    Raises:
        httpx.HTTPError: On transport failures or non-2xx responses.
        RuntimeError: If the response contains no choices.
    """
    payload = {
        "model": spec.model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    {"type": "text", "text": spec.prompt},
                ],
            }
        ],
        "max_tokens": spec.max_tokens,
        "temperature": spec.temperature,
    }
    if spec.extra_body:
        payload.update(spec.extra_body)

    resp = await client.post(
        f"{spec.base_url}/v1/chat/completions",
        json=payload,
        timeout=settings.ocr_request_timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Backend '{spec.name}' returned no choices: {data}")
    return choices[0]["message"]["content"] or ""


async def ocr_pages(
    client: httpx.AsyncClient, spec: ModelSpec, images: list[Path]
) -> str:
    """OCR one or more page images and join them into a single Markdown string.

    Args:
        client: Shared async HTTP client.
        spec: The backend model specification.
        images: Ordered page images.

    Returns:
        Combined Markdown; multiple pages are separated by a ``---`` rule.
    """
    parts: list[str] = []
    for idx, image in enumerate(images, start=1):
        logger.info("OCR %s page %d/%d (%s)", spec.name, idx, len(images), image.name)
        parts.append(await ocr_image(client, spec, image))
    return "\n\n---\n\n".join(parts)
