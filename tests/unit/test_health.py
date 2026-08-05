"""Health-check endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_platform import __version__


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "version": __version__}


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"].startswith("Enterprise Multi-Agent")
