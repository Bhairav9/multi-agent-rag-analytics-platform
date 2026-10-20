"""Unit tests for LLM provider (hermetic — no Ollama server required)."""

from __future__ import annotations

import asyncio
import pytest
from pydantic import ValidationError
from unittest.mock import patch

from rag_platform.llm.provider import OllamaProvider, get_llm_provider


class FakeChunk:
    def __init__(self, content: str) -> None:
        self.content = content


async def fake_astream(_messages):
    for token in ["Hello", " world", "!"]:
        yield FakeChunk(token)


async def _collect(provider: OllamaProvider, prompt: str) -> list[str]:
    return [t async for t in provider.astream(prompt)]


def test_ollama_provider_streams_tokens() -> None:
    with patch("rag_platform.llm.provider.ChatOllama") as MockChatOllama:
        mock_llm = MockChatOllama.return_value
        mock_llm.astream = fake_astream
        provider = OllamaProvider(model="llama3.1:8b-instruct", base_url="http://localhost:11434")
        tokens = asyncio.run(_collect(provider, "hi"))

    assert tokens == ["Hello", " world", "!"]


def test_get_llm_provider_returns_ollama(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b-instruct")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    from rag_platform.config import get_settings
    get_settings.cache_clear()

    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.1:8b-instruct"

    get_settings.cache_clear()


def test_unknown_provider_raises(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    from rag_platform.config import get_settings
    get_settings.cache_clear()

    with pytest.raises((ValueError, ValidationError)):
        get_llm_provider()

    get_settings.cache_clear()
