#!/usr/bin/env python3
"""Send a file to a running OCR backend and write the recognized Markdown.

Usage:
    # start one backend first, e.g.:  docker compose up deepseek-ocr
    python cli.py deepseek-ocr data/scan.pdf
    python cli.py qwen3-vl-8b page.png --out output --dpi 300
    python cli.py hunyuan-ocr doc.pdf --url http://localhost:8002

Talks directly to the backend's vLLM OpenAI API — no gateway. Only the model
whose container is currently running can be reached.
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

from inputs import InputError, load_images
from ocr_client import ocr_pages
from outputs import write_markdown
from registry import MODELS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", choices=sorted(MODELS), help="Which OCR model to use."
    )
    parser.add_argument("file", type=Path, help="Image or PDF to OCR.")
    parser.add_argument(
        "--url",
        default=None,
        help="Backend base URL. Default: http://<host>:<model port>.",
    )
    parser.add_argument(
        "--host", default="localhost", help="Host for the default URL (default: localhost)."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("output"), help="Output root dir (default: output)."
    )
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI (default: 200).")
    parser.add_argument("--max-pages", type=int, default=100, help="Max PDF pages (default: 100).")
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="Per-page request timeout, s (default: 600)."
    )
    args = parser.parse_args()

    spec = MODELS[args.model]
    url = args.url or spec.default_url(args.host)

    try:
        images, temp_paths = load_images(args.file, dpi=args.dpi, max_pages=args.max_pages)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"model={spec.name} url={url} file={args.file} pages={len(images)}", flush=True)
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=args.timeout) as client:
            try:
                health = client.get(f"{url}/health", timeout=5.0)
                ready = health.status_code == 200
            except httpx.HTTPError:
                ready = False
            if ready:
                print("backend: ready (/health 200)", flush=True)
            else:
                print(
                    "backend: not ready yet — model is still loading; "
                    "waiting (check `docker compose logs -f`) ...",
                    flush=True,
                )
            markdown = ocr_pages(client, url, spec, images)
    except httpx.ConnectError:
        print(
            f"error: cannot reach backend at {url}. Is the '{spec.name}' container running?",
            file=sys.stderr,
        )
        return 1
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        for tmp in temp_paths:
            tmp.unlink(missing_ok=True)

    out_path = write_markdown(args.out, spec.name, args.file, markdown)
    elapsed = time.perf_counter() - t0
    print(f"ok: wrote {out_path} ({len(markdown)} chars, {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
