"""LLM provider abstraction — OllamaProvider + factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from rag_platform.config import get_settings


class LLMProvider(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    model: str

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """Yield response tokens as they are generated."""
        ...


class OllamaProvider:
    """LLM provider backed by a local Ollama server."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self._llm = ChatOllama(model=model, base_url=base_url)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self._llm.astream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield str(chunk.content)


def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider based on settings."""
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.llm_model, settings.ollama_base_url)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
