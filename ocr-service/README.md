# OCR Service

Local HTTP API for extracting text from images and PDF documents, powered by **PaddleOCR (PaddleX v3)** and served via **FastAPI**.

Optimised for **NVIDIA RTX 3050 8 GB** — uses the lightweight PP-OCRv5 *mobile* models to keep VRAM consumption low.

---

## Project layout

```
ocr-service/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes
│   ├── ocr_engine.py    # PaddleOCR singleton + async GPU lock
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

### `POST /ocr/image` — single image

Accepts: `png`, `jpg`, `jpeg`, `webp`, `bmp`

```bash
curl -X POST "http://localhost:6666/ocr/image" \
     -F "file=@page.png"
```

Response:
```json
{
  "filename": "page.png",
  "text": "Распознанный текст...",
  "lines": ["Строка 1", "Строка 2"],
  "engine": "paddleocr",
  "language": "ru"
}
```

### `POST /ocr/pdf` — PDF document

```bash
curl -X POST "http://localhost:6666/ocr/pdf" \
     -F "file=@document.pdf"
```

Response:
```json
{
  "filename": "document.pdf",
  "pages": [
    {"page": 1, "text": "...", "lines": ["..."]},
    {"page": 2, "text": "...", "lines": ["..."]}
  ],
  "full_text": "Объединённый текст всех страниц..."
}
```

### `POST /ocr/batch` — multiple images

```bash
curl -X POST "http://localhost:6666/ocr/batch" \
     -F "files=@img1.png" \
     -F "files=@img2.jpg" \
     -F "files=@img3.webp"
```

Response:
```json
{
  "results": [
    {"filename": "img1.png", "text": "...", "lines": [...], "engine": "paddleocr", "language": "ru", "error": null},
    ...
  ]
}
```

---

## Sample Python client

```bash
# Single image
python sample_client.py image page.png

# PDF
python sample_client.py pdf document.pdf

# Batch
python sample_client.py batch img1.png img2.jpg

# Custom URL
python sample_client.py --url http://192.168.1.10:6666 image scan.png
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

> The test suite mocks the OCR engine — no GPU or PaddleOCR installation is required for `/health`.

---

## Design decisions

| Concern | Decision |
|---|---|
| GPU VRAM (8 GB RTX 3050) | PP-OCRv5 *mobile* models (~100 MB combined); single uvicorn worker |
| Concurrent GPU access | `asyncio.Lock` in `ocr_engine.py` — OCR calls are serialised |
| Temporary files | `tempfile.NamedTemporaryFile`; deleted in `finally` blocks |
| Model persistence | Docker named volume `paddleocr-cache` → `/root/.paddlex` |
| First-request latency | Models pre-downloaded during `docker build` |
