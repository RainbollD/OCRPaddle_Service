from pathlib import Path


def write_markdown(out_dir: Path, model: str, input_path: Path, markdown: str) -> Path:
    """Write OCR output to ``<out_dir>/<model>/<input-stem>.md``.

    Args:
        out_dir: Root output directory.
        model: Model name (used as the output subfolder).
        input_path: The resolved input file path (its stem names the .md file).
        markdown: The Markdown text to write.

    Returns:
        The path to the written Markdown file.
    """
    target_dir = out_dir / model
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{input_path.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path
