"""Unit tests for ingestion pipeline (hermetic — mocked SEC + fake embedders)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import SparseVector

from rag_platform.ingestion.pipeline import run_pipeline
from rag_platform.ingestion.sec_edgar import FilingMeta

FAKE_FILING = FilingMeta(
    ticker="AAPL",
    cik="0000320193",
    form="10-K",
    accession="0000320193-24-000123",
    primary_document="aapl-10k.htm",
    filing_date="2024-10-30",
)

FIXTURE_TEXT = """
ITEM 1. BUSINESS
Apple Inc. designs and markets smartphones worldwide.

ITEM 1A. RISK FACTORS
Supply chain disruptions may adversely affect operations.
"""


class FakeDense:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeSparse:
    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 5], values=[0.9, 0.4]) for _ in texts]


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    filing_dir = tmp_path / "AAPL"
    filing_dir.mkdir(parents=True)
    filing_path = filing_dir / "0000320193-24-000123.html"
    filing_path.write_text(FIXTURE_TEXT, encoding="utf-8")
    return tmp_path


def test_run_pipeline_indexes_chunks(raw_dir: Path) -> None:
    with (
        patch("rag_platform.ingestion.pipeline.lookup_cik", return_value="0000320193"),
        patch("rag_platform.ingestion.pipeline.list_filings", return_value=[FAKE_FILING]),
        patch("rag_platform.ingestion.pipeline.download_filing", return_value=raw_dir / "AAPL" / "0000320193-24-000123.html"),
        patch("rag_platform.ingestion.pipeline.FastEmbedDense", return_value=FakeDense()),
        patch("rag_platform.ingestion.pipeline.FastEmbedSparse", return_value=FakeSparse()),
        patch("rag_platform.ingestion.pipeline.QdrantClient") as mock_qdrant,
    ):
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_client.collection_exists.return_value = True

        n_filings, n_chunks = run_pipeline(["AAPL"], years=(2024,), raw_dir=raw_dir)

    assert n_filings == 1
    assert n_chunks >= 2


def test_run_pipeline_skips_empty_filing(raw_dir: Path) -> None:
    empty_path = raw_dir / "AAPL" / "empty.html"
    empty_path.write_text("", encoding="utf-8")

    with (
        patch("rag_platform.ingestion.pipeline.lookup_cik", return_value="0000320193"),
        patch("rag_platform.ingestion.pipeline.list_filings", return_value=[FAKE_FILING]),
        patch("rag_platform.ingestion.pipeline.download_filing", return_value=empty_path),
        patch("rag_platform.ingestion.pipeline.FastEmbedDense", return_value=FakeDense()),
        patch("rag_platform.ingestion.pipeline.FastEmbedSparse", return_value=FakeSparse()),
        patch("rag_platform.ingestion.pipeline.QdrantClient") as mock_qdrant,
    ):
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_client.collection_exists.return_value = True

        n_filings, n_chunks = run_pipeline(["AAPL"], years=(2024,), raw_dir=raw_dir)

    assert n_filings == 0
    assert n_chunks == 0
