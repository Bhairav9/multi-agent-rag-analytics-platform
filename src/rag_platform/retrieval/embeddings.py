from collections.abc import Iterable
from typing import Protocol

from fastembed import TextEmbedding


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class DenseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class _EmbeddingModel(Protocol):
    def embed(self, documents: list[str]) -> Iterable[Iterable[float]]:
        ...


class FastEmbedDense:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        model: _EmbeddingModel | None = None,
    ) -> None:
        self._model = model if model is not None else TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        return [
            [float(value) for value in vector]
            for vector in self._model.embed(texts)
        ]
    