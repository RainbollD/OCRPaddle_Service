from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the OCR gateway.

    Values are read from environment variables (see ``docker-compose.yml``) and
    fall back to the defaults below for local runs.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Filesystem mounts shared with the host (and visible inside the container).
    data_dir: Path = Path("/data")
    output_dir: Path = Path("/output")

    # Docker network the gateway and backends share. Backends are reachable by
    # their compose service name on this network.
    docker_network: str = "ocr-net"

    # How long to wait for a backend to become healthy after starting it
    # (first start includes model download + load and can take minutes).
    backend_ready_timeout_s: int = 900
    backend_poll_interval_s: float = 3.0

    # Per-request OCR generation timeout (one page can take tens of seconds).
    ocr_request_timeout_s: float = 600.0

    # PDF rendering resolution and page cap.
    pdf_dpi: int = 200
    max_pdf_pages: int = 100

    ALLOWED_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


settings = Settings()
