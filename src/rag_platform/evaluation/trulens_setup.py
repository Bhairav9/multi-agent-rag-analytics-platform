"""TruLens groundedness evaluator — LLM-as-judge pattern."""

from __future__ import annotations

import statistics

from trulens.core import TruSession
from trulens.providers.langchain import Langchain

from rag_platform.llm.provider import get_llm_provider


class GroundednessEvaluator:
    """Scores answer groundedness against source chunks using TruLens."""

    def __init__(self) -> None:
        self._session = TruSession()
        llm_provider = get_llm_provider()
        self._provider = Langchain(chain=llm_provider._llm)

    def score(self, answer: str, chunks: list[str]) -> float:
        """Return groundedness score 0.0–1.0 for one answer."""
        context = "\n---\n".join(chunks)
        result = self._provider.groundedness_measure_with_cot_reasons(
            source=context, statement=answer
        )
        return float(result[0]) if isinstance(result, tuple) else float(result)

    def evaluate_batch(
        self, pairs: list[tuple[str, list[str]]]
    ) -> list[float]:
        """Score a batch of (answer, chunks) pairs."""
        return [self.score(answer, chunks) for answer, chunks in pairs]

    @staticmethod
    def mean(scores: list[float]) -> float:
        return statistics.mean(scores) if scores else 0.0


        