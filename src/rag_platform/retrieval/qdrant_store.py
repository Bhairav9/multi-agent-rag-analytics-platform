"""Qdrant collection management — dense + sparse named vectors, one point per chunk."""

from __future__ import annotations

import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

_ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def ensure_collection(self, dense_dim: int = 384) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
            )
        # Cloud Qdrant requires explicit payload index for filtered searches.
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="ticker",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    @staticmethod
    def point_id(*parts: str) -> str:
        """Deterministic ID so re-ingesting the same chunk overwrites instead of duplicating."""
        return str(uuid.uuid5(_ID_NAMESPACE, ":".join(parts)))

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        payloads: list[dict[str, object]],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
    ) -> None:
        if not (
            len(ids) == len(texts) == len(payloads) == len(dense_vectors) == len(sparse_vectors)
        ):
            raise ValueError(
                "ids, texts, payloads, dense_vectors, sparse_vectors must be the same length"
            )

        points = [
            PointStruct(
                id=ids[i],
                vector={DENSE_VECTOR_NAME: dense_vectors[i], SPARSE_VECTOR_NAME: sparse_vectors[i]},
                payload={**payloads[i], "text": texts[i]},
            )
            for i in range(len(ids))
        ]
        batch_size = 10
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            for attempt in range(3):
                try:
                    self._client.upsert(collection_name=self._collection, points=batch)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)

    def search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        top_k: int = 10,
        prefetch_limit: int = 20,
        query_filter=None,
    ) -> list[dict[str, object]]:
        result = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=prefetch_limit),
                Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=prefetch_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [{"score": point.score, **(point.payload or {})} for point in result.points]
