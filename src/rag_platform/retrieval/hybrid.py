"""Public hybrid retrieval facade — embeds a query and runs RRF fusion search."""

from __future__ import annotations

from dataclasses import dataclass

from rag_platform.retrieval.bm25 import SparseEmbedder
from rag_platform.retrieval.embeddings import DenseEmbedder
from rag_platform.retrieval.qdrant_store import QdrantStore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


class HybridRetriever:
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        store: QdrantStore,
    ) -> None:
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        dense_vector = self._dense.embed([query])[0]
        sparse_vector = self._sparse.embed([query])[0]
        hits = self._store.search(dense_vector, sparse_vector, top_k=top_k)
        return [
            RetrievedChunk(
                text=hit.pop("text"),
                score=hit.pop("score"),
                metadata=hit
            )
            for hit in hits
        ]        