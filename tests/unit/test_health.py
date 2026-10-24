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


def test_healthz_ready_requires_api_key(client: TestClient) -> None:
    response = client.get("/healthz/ready")
    assert response.status_code == 401


def test_healthz_ready_accepts_valid_api_key(client: TestClient) -> None:
    # In test mode Redis and Qdrant are not initialised → 503 degraded, not 401.
    response = client.get("/healthz/ready", headers={"X-API-Key": "change-me"})
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
