from pathlib import Path
import asyncio
import cv2
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""


def _read_image_unicode(path: Path) -> np.ndarray:
    """
    Read an image from disk with Unicode path support.

    Args:
        path: Path to the input image.

    Returns:
        Loaded image as a NumPy array in BGR format.

    Raises:
        ImageProcessingError: If the file cannot be read or decoded.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.exception("Failed to read file: %s", path)
        raise ImageProcessingError(f"Failed to read file: {path}") from exc

    if image is None:
        logger.error("File is not a valid image: %s", path)
        raise ImageProcessingError(f"File is not a valid image: {path}")

    return image


def _write_image_unicode(path: Path, image: np.ndarray) -> None:
    """
    Save an image to disk with Unicode path support.

    Args:
        path: Output file path.
        image: Image to save as a NumPy array.

    Raises:
        ImageProcessingError: If the image cannot be encoded or saved.
    """
    ext = path.suffix.lower()

    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"]:
        ext = ".png"
        path = path.with_suffix(ext)

    success, encoded = cv2.imencode(ext, image)

    if not success:
        logger.error("Failed to encode image: %s", path)
        raise ImageProcessingError(f"Failed to encode image: {path}")

    try:
        encoded.tofile(str(path))
    except Exception as exc:
        logger.exception("Failed to save file: %s", path)
        raise ImageProcessingError(f"Failed to save file: {path}") from exc


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct a small text skew angle in a binary image.

    Args:
        image: Binary image, usually black text on white background.

    Returns:
        Deskewed image. If the skew angle is too small or cannot be detected,
        the original image is returned.
    """
    inverted = cv2.bitwise_not(image)
    coords = np.column_stack(np.where(inverted > 0))

    if len(coords) < 50:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.3:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


def _improve_image_for_ocr_sync(
    image_path: str | Path,
    scale: float = 2.0,
    output_suffix: str = "_ocr",
    overwrite: bool = True,
) -> Path:
    """
    Synchronous OCR image preprocessing implementation.

    This function is intentionally synchronous because OpenCV operations are
    blocking. The async wrapper runs this function in a separate thread.

    Args:
        image_path: Path to the source image.
        scale: Resize factor. Values greater than 1 increase image resolution.
        output_suffix: Suffix added to the processed file name.
        overwrite: If True, overwrite the original file.

    Returns:
        Path to the saved processed image.

    Raises:
        FileNotFoundError: If the input image does not exist.
        ImageProcessingError: If the input path is invalid or processing fails.
    """
    input_path = Path(image_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if not input_path.is_file():
        logger.error("Path is not a file: %s", input_path)
        raise ImageProcessingError(f"Path is not a file: {input_path}")

    image = _read_image_unicode(input_path)

    # Increase image resolution to make small text easier to recognize.
    if scale and scale != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # Convert the image to grayscale because OCR usually does not need color.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise while preserving text edges.
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=30,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # Improve local contrast, especially useful for uneven lighting.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    contrast = clahe.apply(denoised)

    # Convert the image to black-and-white using adaptive thresholding.
    binary = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    # Correct small text rotation.
    result = _deskew(binary)

    # Apply light morphological cleanup to strengthen text shapes.
    kernel = np.ones((1, 1), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    if overwrite:
        output_path = input_path
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}{output_suffix}.png"
        )

    _write_image_unicode(output_path, result)

    return output_path


async def improve_image_for_ocr(
    image_path: str | Path,
    scale: float = 2.0,
    output_suffix: str = "_ocr",
    overwrite: bool = True,
) -> Path:
    """
    Asynchronously improve image quality for OCR and save the processed image.

    OpenCV operations are blocking, so this function offloads the synchronous
    image processing pipeline to a worker thread using asyncio.to_thread.

    Args:
        image_path: Path to the source image.
        scale: Resize factor. Values greater than 1 increase image resolution.
        output_suffix: Suffix added to the processed file name.
        overwrite: If True, overwrite the original file.

    Returns:
        Path to the saved processed image.
    """
    return await asyncio.to_thread(
        _improve_image_for_ocr_sync,
        image_path,
        scale,
        output_suffix,
        overwrite,
    )