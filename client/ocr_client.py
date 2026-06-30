import base64
import mimetypes
import time
from pathlib import Path

import httpx

from registry import ModelSpec


def _image_data_url(path: Path) -> str:
    """Read an image and encode it as a base64 ``data:`` URL."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def ocr_image(client: httpx.Client, url: str, spec: ModelSpec, image: Path) -> str:
    """Run OCR on a single image via a backend's OpenAI-compatible API.

    Args:
        client: An open HTTP client.
        url: Backend base URL (e.g. ``http://localhost:8001``).
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
    resp = client.post(f"{url}/v1/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Backend returned no choices: {data}")
    return choices[0]["message"]["content"] or ""


def ocr_pages(
    client: httpx.Client, url: str, spec: ModelSpec, images: list[Path]
) -> str:
    """OCR one or more page images and join them into a single Markdown string.

    Multiple pages are separated by a ``---`` rule.
    """
    parts: list[str] = []
    total = len(images)
    for idx, image in enumerate(images, start=1):
        print(f"  page {idx}/{total}: sending ...", flush=True)
        t0 = time.perf_counter()
        text = ocr_image(client, url, spec, image)
        dt = time.perf_counter() - t0
        print(f"  page {idx}/{total}: done ({len(text)} chars, {dt:.1f}s)", flush=True)
        parts.append(text)
    return "\n\n---\n\n".join(parts)
