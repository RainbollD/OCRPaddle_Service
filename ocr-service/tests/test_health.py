"""Minimal smoke-test for the /health endpoint.

Run with:
    pytest tests/test_health.py -v

The test patches out the OCR engine initialisation so it does not require
PaddleOCR or a GPU to be installed in the test environment.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Return a TestClient with OCR engine mocked out."""
    mock_ocr = MagicMock()

    with patch("app.ocr_engine.init_ocr", return_value=None), \
         patch("app.ocr_engine._ocr_instance", mock_ocr):
        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


def test_health_content_type(client: TestClient) -> None:
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
