"""Unit tests for QdrantStore using an in-memory Qdrant engine."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag_platform.retrieval.qdrant_store import QdrantStore


def _make_store() -> QdrantStore:
    store = QdrantStore(QdrantClient(location=":memory:"), "test_collection")
    store.ensure_collection(dense_dim=4)
    return store


def test_upsert_and_search_roundtrip() -> None:
    store = _make_store()

    store.upsert(
        ids=[store.point_id("AAPL", "acc-1", "0")],
        texts=["Apple's risk factors include supply chain disruption."],
        payloads=[{"ticker": "AAPL", "item": "1A"}],
        dense_vectors=[[0.1, 0.2, 0.3, 0.4]],
        sparse_vectors=[SparseVector(indices=[1, 5], values=[0.9, 0.4])],
    )

    hits = store.search(
        dense_vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=SparseVector(indices=[1, 5], values=[0.9, 0.4]),
        top_k=1,
    )

    assert len(hits) == 1
    assert hits[0]["ticker"] == "AAPL"
    assert hits[0]["text"] == "Apple's risk factors include supply chain disruption."


def test_ensure_collection_is_idempotent() -> None:
    store = _make_store()

    store.ensure_collection(dense_dim=4)  # second call must not raise or reset data

    store.upsert(
        ids=[store.point_id("MSFT", "acc-2", "0")],
        texts=["Azure revenue grew."],
        payloads=[{"ticker": "MSFT"}],
        dense_vectors=[[0.5, 0.5, 0.5, 0.5]],
        sparse_vectors=[SparseVector(indices=[2], values=[1.0])],
    )
    hits = store.search(dense_vector=[0.5, 0.5, 0.5, 0.5], sparse_vector=SparseVector(indices=[2], values=[1.0]))
    assert hits[0]["ticker"] == "MSFT"


def test_point_id_is_deterministic() -> None:
    store = _make_store()

    first = store.point_id("AAPL", "acc-1", "0")
    second = store.point_id("AAPL", "acc-1", "0")
    different = store.point_id("AAPL", "acc-1", "1")

    assert first == second
    assert first != different


def test_repeated_upsert_overwrites_not_duplicates() -> None:
    store = _make_store()
    point_id = store.point_id("AAPL", "acc-1", "0")

    for text in ("first version", "updated version"):
        store.upsert(
            ids=[point_id],
            texts=[text],
            payloads=[{"ticker": "AAPL"}],
            dense_vectors=[[0.1, 0.2, 0.3, 0.4]],
            sparse_vectors=[SparseVector(indices=[1], values=[1.0])],
        )

    hits = store.search(dense_vector=[0.1, 0.2, 0.3, 0.4], sparse_vector=SparseVector(indices=[1], values=[1.0]))
    assert len(hits) == 1
    assert hits[0]["text"] == "updated version"


def test_upsert_raises_on_mismatched_lengths() -> None:
    store = _make_store()

    with pytest.raises(ValueError, match="same length"):
        store.upsert(
            ids=["id-1", "id-2"],
            texts=["only one text"],
            payloads=[{"ticker": "AAPL"}],
            dense_vectors=[[0.1, 0.2, 0.3, 0.4]],
            sparse_vectors=[SparseVector(indices=[1], values=[1.0])],
        )