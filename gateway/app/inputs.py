import logging
import tempfile
from pathlib import Path

import fitz

from app.config import settings

logger = logging.getLogger(__name__)


class InputError(Exception):
    """Raised when an input path is missing, outside the data dir, or unsupported."""


def resolve_input_path(rel_path: str) -> Path:
    """Resolve a request path against the data dir and guard against escapes.

    Args:
        rel_path: Path from the request, relative to ``settings.data_dir``.
            Absolute paths that already point inside the data dir are accepted.

    Returns:
        The resolved, existing file path inside the data dir.

    Raises:
        InputError: If the path escapes the data dir or does not exist.
    """
    data_root = settings.data_dir.resolve()
    candidate = Path(rel_path)
    if not candidate.is_absolute():
        candidate = data_root / candidate
    resolved = candidate.resolve()

    if not (resolved == data_root or data_root in resolved.parents):
        raise InputError(
            f"Path '{rel_path}' resolves outside the data directory."
        )
    if not resolved.is_file():
        raise InputError(f"File not found: '{rel_path}'.")
    return resolved


def render_to_images(path: Path) -> tuple[list[Path], list[Path]]:
    """Turn an input file into a list of image paths ready for OCR.

    Images are returned as-is. PDFs are rendered to temporary PNGs (one per
    page). The caller must delete any returned temporary files.

    Args:
        path: Resolved input file path.

    Returns:
        A tuple ``(image_paths, temp_paths)`` where ``image_paths`` is the
        ordered list of images to OCR and ``temp_paths`` is the subset that the
        caller is responsible for deleting.

    Raises:
        InputError: If the file extension is neither a supported image nor PDF.
    """
    ext = path.suffix.lower()
    if ext in settings.ALLOWED_IMAGE_EXTENSIONS:
        return [path], []
    if ext == ".pdf":
        rendered = _pdf_to_images(path, settings.pdf_dpi, settings.max_pdf_pages)
        return rendered, rendered

    allowed = ", ".join(sorted(settings.ALLOWED_IMAGE_EXTENSIONS)) + ", .pdf"
    raise InputError(f"Unsupported file type '{ext}'. Allowed: {allowed}.")


def _pdf_to_images(pdf_path: Path, dpi: int, max_pages: int) -> list[Path]:
    """Render every page of a PDF to temporary PNG files.

    Landscape pages are prerotated so OCR receives upright page images. Ported
    from the previous PaddleX service's ``pdf_utils.pdf_to_images``.

    Args:
        pdf_path: Path to the source PDF.
        dpi: Render resolution in dots per inch.
        max_pages: Maximum number of pages accepted before rendering starts.

    Returns:
        Ordered temporary PNG paths, one per PDF page.

    Raises:
        InputError: If the PDF has more pages than ``max_pages``.
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    image_paths: list[Path] = []
    doc = fitz.open(str(pdf_path))
    try:
        total_pages = len(doc)
        if total_pages > max_pages:
            raise InputError(
                f"PDF has {total_pages} pages; maximum allowed is {max_pages}."
            )
        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            rect = page.rect
            if rect.width > rect.height:
                pix = page.get_pixmap(matrix=matrix.prerotate(90), alpha=False)
            else:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, prefix=f"ocr_pdf_p{page_index + 1}_"
            )
            tmp.close()
            pix.save(tmp.name)
            image_paths.append(Path(tmp.name))
            logger.debug("Rendered PDF page %d -> %s", page_index + 1, tmp.name)
    finally:
        doc.close()
    return image_paths
