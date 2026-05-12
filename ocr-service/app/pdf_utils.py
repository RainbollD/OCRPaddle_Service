import logging
import tempfile
from pathlib import Path

import fitz

from app.config import settings

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: Path, dpi: int | None = None) -> list[Path]:
    """Render every page of *pdf_path* to a PNG file.

    Returns an ordered list of temporary PNG paths.
    The caller is responsible for deleting these files.
    """
    dpi = dpi or settings.pdf_dpi
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    image_paths: list[Path] = []
    doc = fitz.open(str(pdf_path))

    try:
        for page_index in range(len(doc)):
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
