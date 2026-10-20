from dataclasses import dataclass

import rag_platform.retrieval.bm25 as bm25
from rag_platform.retrieval.bm25 import (
    DEFAULT_SPARSE_MODEL,
    FastEmbedSparse,
)


@dataclass
class FakeSparseResult:
    indices: list[int]
    values: list[int | float]


class FakeSparseModel:
    def __init__(self, results: list[FakeSparseResult]) -> None:
        self.results = results
        self.calls: list[list[str]] = []

    def embed(self, documents: list[str]) -> list[FakeSparseResult]:
        self.calls.append(documents)
        return self.results


def test_embed_returns_qdrant_sparse_vectors_in_order() -> None:
    model = FakeSparseModel(
        [
            FakeSparseResult(indices=[1, 4], values=[2, 0.5]),
            FakeSparseResult(indices=[3], values=[1]),
        ]
    )
    embedder = FastEmbedSparse(model=model)

    result = embedder.embed(["first", "second"])

    assert len(result) == 2
    assert result[0].indices == [1, 4]
    assert result[0].values == [2.0, 0.5]
    assert result[1].indices == [3]
    assert result[1].values == [1.0]
    assert model.calls == [["first", "second"]]

def test_embed_empty_input_does_not_call_model() -> None:
    model = FakeSparseModel(results=[])
    embedder = FastEmbedSparse(model=model)

    result = embedder.embed([])

    assert result == []
    assert model.calls == []


def test_default_model_name_is_passed_to_fastembed(monkeypatch) -> None:
    created_with: list[str] = []

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            created_with.append(model_name)

        def embed(self, documents: list[str]) -> list[FakeSparseResult]:
            return [FakeSparseResult(indices=[7], values=[1]) for _ in documents]

    monkeypatch.setattr(bm25, "SparseTextEmbedding", FakeTextEmbedding)

    embedder = FastEmbedSparse()

    assert created_with == [DEFAULT_SPARSE_MODEL]
    assert embedder.embed(["hello"])[0].indices == [7]