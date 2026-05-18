import asyncio
from logging import getLogger
from pathlib import Path

from PIL import Image

logger = getLogger(__name__)

MAX_OCR_IMAGE_SIDE = 3500
MAX_UPSCALE_FACTOR = 2.0


def increasing_image_quality(image_path: Path) -> Path:
    """Upscale an image in place to improve OCR recognition quality.

    The service stores uploads and rendered PDF pages as temporary files. This
    helper adjusts image resolution with a high-quality Lanczos filter while
    keeping the largest side comfortably below PaddleOCR's 4000px limit. Small
    images are upscaled, oversized images are downscaled, and the same path is
    returned so the OCR pipeline can keep working with the temporary file it
    already owns.

    Args:
        image_path (Path): Path to an existing image file.

    Returns:
        Path: The same path after the image has been rewritten with a better
            OCR-friendly resolution.

    Raises:
        FileNotFoundError: If the provided image path does not exist.
        PIL.UnidentifiedImageError: If Pillow cannot identify the image file.
        OSError: If the image cannot be read or written.
    """
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as img:
        scale = min(MAX_UPSCALE_FACTOR, MAX_OCR_IMAGE_SIDE / max(img.width, img.height))
        if scale == 1:
            logger.debug("Image already at optimal OCR size, skipping resize: %s", image_path)
            return image_path

        new_size = (round(img.width * scale), round(img.height * scale))
        # Lanczos keeps text edges sharper than cheaper interpolation filters.
        upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
        upscaled.save(image_path)

    logger.debug(
        "Resized image for OCR preprocessing: %s scale=%.2f size=%s",
        image_path,
        scale,
        new_size,
    )

    return image_path


async def improve_image_for_ocr(image_path: str | Path) -> Path:
    """Improve image quality for OCR without blocking the event loop.

    The synchronous Pillow work is executed in a worker thread so the FastAPI
    event loop can continue serving other requests.

    Args:
        image_path (str | Path): Path to an existing image file.

    Returns:
        Path: The same path after OCR preprocessing has been applied.

    Raises:
        FileNotFoundError: If the provided image path does not exist.
        PIL.UnidentifiedImageError: If Pillow cannot identify the image file.
        OSError: If the image cannot be read or written.
    """
    return await asyncio.to_thread(increasing_image_quality, Path(image_path))
