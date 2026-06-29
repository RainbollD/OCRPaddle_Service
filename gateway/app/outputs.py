import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def write_markdown(model: str, input_path: Path, markdown: str) -> Path:
    """Write OCR output to ``output/<model>/<input-stem>.md``.

    Args:
        model: Model name (used as the output subfolder).
        input_path: The resolved input file path (its stem names the .md file).
        markdown: The Markdown text to write.

    Returns:
        The path to the written Markdown file.
    """
    out_dir = settings.output_dir / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote %s (%d chars).", out_path, len(markdown))
    return out_path
