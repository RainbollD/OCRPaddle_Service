FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_MODEL_SOURCE=BOS \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    OCR_LANG=ru \
    OCR_DEVICE=gpu:0 \
    MAX_UPLOAD_MB=50 \
    PDF_DPI=200

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libpng-dev \
        libjpeg-dev \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /opt/ocr-service

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install paddlepaddle-gpu==3.0.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ && \
    python3 -m pip install -r requirements.txt

COPY app/ ./app/

# Copy PaddleOCR/PaddleX model archives from the local build context.
# During docker build, NVIDIA driver libraries such as libcuda.so.1 are not
# mounted into the build container, so instantiating PaddleOCR with the GPU
# wheel fails even if device="cpu".
COPY models/PP-OCRv3_mobile_det_infer.tar \
     models/eslav_PP-OCRv5_mobile_rec_infer.tar \
     /tmp/models/

RUN set -eux; \
    MODEL_ROOT="/root/.paddlex/official_models"; \
    mkdir -p "${MODEL_ROOT}"; \
    for MODEL in PP-OCRv3_mobile_det eslav_PP-OCRv5_mobile_rec; do \
        mkdir -p "${MODEL_ROOT}/${MODEL}"; \
        tar -xf "/tmp/models/${MODEL}_infer.tar" -C "${MODEL_ROOT}/${MODEL}" --strip-components=1; \
    done

RUN mkdir -p /opt/paddleocr-cache && \
    cp -a /root/.paddlex/. /opt/paddleocr-cache/ 2>/dev/null || true

RUN cat > /usr/local/bin/docker-entrypoint.sh <<'EOF' && chmod +x /usr/local/bin/docker-entrypoint.sh
#!/usr/bin/env sh
set -eu

mkdir -p /root/.paddlex
if [ -d /opt/paddleocr-cache ] && [ -z "$(ls -A /root/.paddlex 2>/dev/null)" ]; then
    cp -a /opt/paddleocr-cache/. /root/.paddlex/ 2>/dev/null || true
fi

exec "$@"
EOF

EXPOSE 6666

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["python3", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "6666", "--workers", "1"]
