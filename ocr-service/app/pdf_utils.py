import logging
import tempfile
from pathlib import Path

import fitz

from app.config import settings

logger = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: Path,
    dpi: int | None = None,
    max_pages: int | None = None,
) -> list[Path]:
    """Render every page of a PDF to temporary PNG files.

    Landscape pages are prerotated before rendering so downstream OCR receives
    upright page images. The caller owns the returned temporary files and must
    delete them after OCR finishes.

    Args:
        pdf_path (Path): Path to the source PDF.
        dpi (int | None, optional): Render resolution in dots per inch.
            Defaults to ``settings.pdf_dpi``.
        max_pages (int | None, optional): Maximum number of pages accepted
            before rendering starts. Defaults to ``None``.

    Returns:
        list[Path]: Ordered temporary PNG paths, one per PDF page.

    Raises:
        ValueError: If the PDF has more pages than ``max_pages``.
        fitz.FileDataError: If PyMuPDF cannot open or parse the PDF.
        RuntimeError: If PyMuPDF fails while rendering a page.
    """
    dpi = dpi or settings.pdf_dpi
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    image_paths: list[Path] = []
    doc = fitz.open(str(pdf_path))

    try:
        total_pages = len(doc)
        if max_pages is not None and total_pages > max_pages:
            raise ValueError(
                f"PDF has {total_pages} pages; maximum allowed is {max_pages}."
            )

        for page_index in range(total_pages):
            page = doc.load_page(page_index)

            rect = page.rect
            if rect.width > rect.height:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom).prerotate(90), alpha=False)
            else:
                pix = page.get_pixmap(matrix=matrix, alpha=False)

            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, prefix=f"ocr_pdf_p{page_index + 1}_"
            )
            tmp.close()
            pix.save(tmp.name)
            image_paths.append(Path(tmp.name))
            logger.debug("Rendered PDF page %d → %s", page_index + 1, tmp.name)
    finally:
        doc.close()

    return image_paths
