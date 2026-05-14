"""Minimal smoke-test for the /health endpoint.

Run with:
    pytest tests/test_health.py -v
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Return a TestClient without loading OCR models."""
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
