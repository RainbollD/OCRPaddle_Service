# OCR Service

Local HTTP API for extracting structured text and tables from images and PDF documents, powered by **PPStructureV3 (PaddleX v3)** and served via **FastAPI**.

Optimised for **NVIDIA RTX 3050 8 GB** — uses the lightweight PP-OCRv5 *mobile* models to keep VRAM consumption low.

---

## Project layout

```
ocr-service/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes
│   ├── ocr_engine.py    # PPStructureV3 singleton + async GPU lock
│   ├── pdf_utils.py     # PyMuPDF PDF → PNG rendering
│   ├── schemas.py       # Pydantic response models
│   └── config.py        # Settings from environment variables
├── tests/
│   └── test_health.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── sample_client.py     # CLI helper for manual testing
└── README.md
```

---

## Quick start

### Prerequisites

| Requirement | Notes |
|---|---|
| Docker ≥ 24 with Compose v2 | `docker compose version` |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | `nvidia-ctk` must be on PATH |
| NVIDIA driver ≥ 525 | `nvidia-smi` should show your GPU |

### 1. Build the image

```bash
cd ocr-service
docker compose build
```

The build step downloads and caches the PP-OCRv5 mobile models inside the image layer so the **first request is instant** (no runtime download).

### 2. Start the service

```bash
docker compose up -d
```

The service is available at **http://localhost:6666**.

### 3. Stop the service

```bash
docker compose down
```

---

## Verifying GPU access inside Docker

```bash
# Check that the container can see the GPU
docker compose exec ocr-service nvidia-smi

# Or run a one-shot check
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 nvidia-smi
```

You should see your RTX 3050 listed with ~8 GB total memory.

---

## API reference

### `GET /health`

```bash
curl http://localhost:6666/health
# {"status":"ok"}
```

### `POST /ocr/image/structured` — single image with layout and tables

Accepts: `png`, `jpg`, `jpeg`, `webp`, `bmp`

```bash
curl -X POST "http://localhost:6666/ocr/image/structured" \
     -F "file=@page.png"
```

Run on CPU instead of GPU:

```bash
curl -X POST "http://localhost:6666/ocr/image/structured" \
     -F "file=@page.png" \
     -F "device=cpu"
```

Allowed `device` values: `cpu`, `gpu:0`. If `device` is omitted, the service uses `OCR_DEVICE` from the environment.

Response:
```json
{
  "filename": "page.png",
  "blocks": [
    {
      "type": "text",
      "text": "Распознанный текст...",
      "table": null
    },
    {
      "type": "table",
      "text": "Колонка 1  Колонка 2\nЗначение 1  Значение 2",
      "table": {
        "rows": [
          {"cells": [{"content": "Колонка 1"}, {"content": "Колонка 2"}]},
          {"cells": [{"content": "Значение 1"}, {"content": "Значение 2"}]}
        ],
        "markdown": "| Колонка 1 | Колонка 2 |\n| --- | --- |\n| Значение 1 | Значение 2 |",
        "html": "<table>...</table>"
      }
    }
  ],
  "text": "Распознанный текст...",
  "markdown": "Распознанный текст...\n\n| Колонка 1 | Колонка 2 |\n| --- | --- |\n| Значение 1 | Значение 2 |",
  "engine": "ppstructurev3",
  "language": "ru"
}
```

### `POST /ocr/document/structured` — PDF document with layout and tables

```bash
curl -X POST "http://localhost:6666/ocr/document/structured" \
     -F "file=@document.pdf"
```

Run PDF OCR on CPU:

```bash
curl -X POST "http://localhost:6666/ocr/document/structured" \
     -F "file=@document.pdf" \
     -F "device=cpu"
```

Response:
```json
{
  "filename": "document.pdf",
  "pages": [
    {"page": 1, "blocks": [...], "text": "...", "markdown": "..."},
    {"page": 2, "blocks": [...], "text": "...", "markdown": "..."}
  ],
  "full_text": "Объединённый текст всех страниц...",
  "full_markdown": "Markdown всех страниц..."
}
```

---

## Sample Python client

```bash
# Structured image
curl -X POST "http://localhost:6666/ocr/image/structured" -F "file=@page.png"

# Structured image on CPU
curl -X POST "http://localhost:6666/ocr/image/structured" -F "file=@page.png" -F "device=cpu"

# Structured PDF
curl -X POST "http://localhost:6666/ocr/document/structured" -F "file=@document.pdf"

# Structured PDF on GPU
curl -X POST "http://localhost:6666/ocr/document/structured" -F "file=@document.pdf" -F "device=gpu:0"
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OCR_LANG` | `ru` | OCR language |
| `OCR_DEVICE` | `gpu:0` | PaddlePaddle device string |
| `MAX_UPLOAD_MB` | `50` | Maximum upload size in megabytes |
| `PDF_DPI` | `200` | Resolution for PDF → image rendering |
| `PADDLE_PDX_MODEL_SOURCE` | `BOS` | Model download source (`BOS` = official bucket) |

All variables can be overridden in `docker-compose.yml` under `environment:` or via a `.env` file placed next to `docker-compose.yml`.

---

## Running tests locally

```bash
# Full project dependencies
pip install -r requirements.txt
pytest tests/ -v

# Or a minimal test-only install without PaddleOCR/GPU packages
pip install fastapi python-multipart pydantic-settings pytest httpx
pytest tests/ -v
```

> The health tests do not load OCR models, so no GPU is required for `/health`.

---

## Design decisions

| Concern | Decision |
|---|---|
| GPU VRAM (8 GB RTX 3050) | Structured OCR only; single PPStructureV3 instance and single uvicorn worker |
| Concurrent GPU access | `asyncio.Lock` in `ocr_engine.py` — OCR calls are serialised |
| Temporary files | `tempfile.NamedTemporaryFile`; deleted in `finally` blocks |
| Model persistence | Docker named volume `paddleocr-cache` → `/root/.paddlex` |
| First-request latency | Models pre-downloaded during `docker build` |
