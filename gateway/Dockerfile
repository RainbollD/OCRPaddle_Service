# Lightweight, CPU-only gateway. It never loads a model itself — it drives the
# GPU backend containers over the Docker socket and calls their HTTP APIs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    OUTPUT_DIR=/output

WORKDIR /opt/gateway

# curl is used by the compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY app/ ./app/

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
