from collections.abc import Iterable
from typing import Protocol

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector


DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class SparseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVector]:
        ...


class _SparseEmbeddingResult(Protocol):
    indices: Iterable[int]
    values: Iterable[float]


class _SparseEmbeddingModel(Protocol):
    def embed(self, documents: list[str]) -> Iterable[_SparseEmbeddingResult]:
        ...


class FastEmbedSparse:
    def __init__(
        self,
        model_name: str = DEFAULT_SPARSE_MODEL,
        *,
        model: _SparseEmbeddingModel | None = None,
    ) -> None:
        self._model = (
            model
            if model is not None
            else SparseTextEmbedding(model_name=model_name)
        )

    def embed(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []

        return [
            SparseVector(
                indices=[int(index) for index in vector.indices],
                values=[float(value) for value in vector.values],
            )
            for vector in self._model.embed(texts)
        ]