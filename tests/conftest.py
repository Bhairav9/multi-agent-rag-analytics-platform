"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

# Must be set before any rag_platform import so the lifespan skips
# heavy service initialization (Qdrant / Ollama / Redis).
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("SEC_EDGAR_USER_AGENT", "Test User test@example.com")

import pytest
from fastapi.testclient import TestClient

from rag_platform.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Fresh FastAPI test client per test."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
