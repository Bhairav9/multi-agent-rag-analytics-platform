from rag_platform.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    FastEmbedDense,
)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, documents: list[str]) -> list[list[int]]:
        self.calls.append(documents)
        return [[1, 2], [3, 4]]


def test_embed_returns_vectors_as_floats_in_input_order() -> None:
    model = FakeEmbeddingModel()
    embedder = FastEmbedDense(model=model)

    result = embedder.embed(["first", "second"])

    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert model.calls == [["first", "second"]]


def test_embed_empty_input_does_not_call_model() -> None:
    model = FakeEmbeddingModel()
    embedder = FastEmbedDense(model=model)

    result = embedder.embed([])

    assert result == []
    assert model.calls == []


def test_default_model_name_is_passed_to_fastembed(monkeypatch) -> None:
    created_with: list[str] = []

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            created_with.append(model_name)

        def embed(self, documents: list[str]) -> list[list[int]]:
            return [[5, 6] for _ in documents]

    monkeypatch.setattr(
        "rag_platform.retrieval.embeddings.TextEmbedding",
        FakeTextEmbedding,
    )

    embedder = FastEmbedDense()

    assert created_with == [DEFAULT_EMBEDDING_MODEL]
    assert embedder.embed(["hello"]) == [[5.0, 6.0]]