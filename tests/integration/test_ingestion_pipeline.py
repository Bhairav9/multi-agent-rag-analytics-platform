"""Integration test — real fastembed models, in-memory Qdrant, saved fixture (no network)."""

from __future__ import annotations

import pytest

from qdrant_client import QdrantClient

from rag_platform.ingestion.chunker import chunk_filing
from rag_platform.retrieval.bm25 import FastEmbedSparse
from rag_platform.retrieval.embeddings import FastEmbedDense
from rag_platform.retrieval.hybrid import HybridRetriever
from rag_platform.retrieval.qdrant_store import QdrantStore

FIXTURE_TEXT = """
ITEM 1. BUSINESS
Apple Inc. designs, manufactures, and markets smartphones, personal computers,
tablets, wearables, and accessories worldwide.

ITEM 1A. RISK FACTORS
The Company's business, reputation, results of operations, financial condition,
and stock price can be affected by a number of factors. Supply chain disruptions
may adversely affect our ability to obtain components at reasonable prices.
"""


@pytest.mark.integration
def test_real_embeddings_roundtrip() -> None:
    chunks = chunk_filing(FIXTURE_TEXT, metadata={"ticker": "AAPL", "accession": "test-001"})
    assert len(chunks) >= 2

    store = QdrantStore(QdrantClient(location=":memory:"), "integration_test")
    dense = FastEmbedDense()
    sparse = FastEmbedSparse()

    store.ensure_collection(dense_dim=384)

    texts = [c.text for c in chunks]
    ids = [store.point_id("AAPL", "test-001", str(i)) for i in range(len(chunks))]

    store.upsert(
        ids=ids,
        texts=texts,
        payloads=[c.metadata for c in chunks],
        dense_vectors=dense.embed(texts),
        sparse_vectors=sparse.embed(texts),
    )

    retriever = HybridRetriever(dense, sparse, store)
    results = retriever.retrieve("supply chain risk", top_k=3)

    assert len(results) >= 1
    assert any("supply chain" in r.text.lower() for r in results)
    assert all(r.metadata["ticker"] == "AAPL" for r in results)
