"""Unit tests for HybridRetriever using fake embedders + in-memory Qdrant."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag_platform.retrieval.hybrid import HybridRetriever, RetrievedChunk
from rag_platform.retrieval.qdrant_store import QdrantStore


class FakeDense:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeSparse:
    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 5], values=[0.9, 0.4]) for _ in texts]


def _store_with_data() -> QdrantStore:
    store = QdrantStore(QdrantClient(location=":memory:"), "test_col")
    store.ensure_collection(dense_dim=4)
    store.upsert(
        ids=[store.point_id("AAPL", "acc-1", "0")],
        texts=["Apple's risk factors include supply chain disruption."],
        payloads=[{"ticker": "AAPL", "item": "1A"}],
        dense_vectors=[[0.1, 0.2, 0.3, 0.4]],
        sparse_vectors=[SparseVector(indices=[1, 5], values=[0.9, 0.4])],
    )
    return store


def test_retrieve_returns_cited_chunks() -> None:
    retriever = HybridRetriever(FakeDense(), FakeSparse(), _store_with_data())
    results = retriever.retrieve("supply chain risk", top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].metadata["ticker"] == "AAPL"
    assert results[0].metadata["item"] == "1A"
    assert "supply chain" in results[0].text.lower()


def test_retrieve_respects_top_k() -> None:
    store = QdrantStore(QdrantClient(location=":memory:"), "test_col")
    store.ensure_collection(dense_dim=4)
    store.upsert(
        ids=[store.point_id("AAPL", "acc-1", str(i)) for i in range(5)],
        texts=[f"Chunk number {i}" for i in range(5)],
        payloads=[{"ticker": "AAPL", "item": "1A"} for _ in range(5)],
        dense_vectors=[[0.1, 0.2, 0.3, 0.4] for _ in range(5)],
        sparse_vectors=[SparseVector(indices=[1, 5], values=[0.9, 0.4]) for _ in range(5)],
    )
    retriever = HybridRetriever(FakeDense(), FakeSparse(), store)
    results = retriever.retrieve("any query", top_k=2)

    assert len(results) == 2


def test_retrieve_empty_store_returns_empty_list() -> None:
    store = QdrantStore(QdrantClient(location=":memory:"), "test_col")
    store.ensure_collection(dense_dim=4)
    retriever = HybridRetriever(FakeDense(), FakeSparse(), store)
    results = retriever.retrieve("anything")

    assert results == []
