import tempfile
from pathlib import Path

import fitz

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class InputError(Exception):
    """Raised when an input file is missing or of an unsupported type."""


def load_images(path: Path, dpi: int = 200, max_pages: int = 100) -> tuple[list[Path], list[Path]]:
    """Turn an input file into a list of image paths ready for OCR.

    Images are returned as-is. PDFs (including scanned ones) are rendered to
    temporary PNGs, one per page. The caller must delete any returned temp files.

    Args:
        path: Input file path (image or PDF).
        dpi: Render resolution for PDFs.
        max_pages: Maximum PDF pages accepted before rendering starts.

    Returns:
        A tuple ``(image_paths, temp_paths)`` where ``temp_paths`` is the subset
        the caller must delete.

    Raises:
        InputError: If the file is missing or its type is unsupported.
    """
    if not path.is_file():
        raise InputError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return [path], []
    if ext == ".pdf":
        rendered = _pdf_to_images(path, dpi, max_pages)
        return rendered, rendered

    allowed = ", ".join(sorted(IMAGE_EXTENSIONS)) + ", .pdf"
    raise InputError(f"Unsupported file type '{ext}'. Allowed: {allowed}.")


def _pdf_to_images(pdf_path: Path, dpi: int, max_pages: int) -> list[Path]:
    """Render every page of a PDF to temporary PNG files.

    Landscape pages are prerotated so OCR receives upright page images.

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
    finally:
        doc.close()
    return image_paths
