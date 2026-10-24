"""LLM provider abstraction — OllamaProvider, AnthropicProvider + factory."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Protocol

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from rag_platform.api.metrics import LLM_LATENCY, LLM_REQUESTS
from rag_platform.config import get_settings


class LLMProvider(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    model: str

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """Yield response tokens as they are generated."""
        ...

    async def ainvoke_messages(self, messages: list) -> str:
        """Send a list of messages and return the full response text."""
        ...


class OllamaProvider:
    """LLM provider backed by a local Ollama server."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self._llm = ChatOllama(model=model, base_url=base_url)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        start = time.monotonic()
        try:
            async for chunk in self._llm.astream([HumanMessage(content=prompt)]):
                if chunk.content:
                    yield str(chunk.content)
        finally:
            LLM_REQUESTS.labels(provider="ollama", model=self.model).inc()
            LLM_LATENCY.labels(provider="ollama", model=self.model).observe(time.monotonic() - start)

    async def ainvoke_messages(self, messages: list) -> str:
        start = time.monotonic()
        try:
            response = await self._llm.ainvoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        finally:
            LLM_REQUESTS.labels(provider="ollama", model=self.model).inc()
            LLM_LATENCY.labels(provider="ollama", model=self.model).observe(time.monotonic() - start)


class OpenAIProvider:
    """LLM provider backed by the OpenAI API via LangChain."""

    def __init__(self, api_key: str, model: str) -> None:
        from langchain_openai import ChatOpenAI  # optional dep
        self.model = model
        self._llm = ChatOpenAI(model=model, api_key=api_key, streaming=True)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        start = time.monotonic()
        try:
            async for chunk in self._llm.astream([HumanMessage(content=prompt)]):
                if chunk.content:
                    yield str(chunk.content)
        finally:
            LLM_REQUESTS.labels(provider="openai", model=self.model).inc()
            LLM_LATENCY.labels(provider="openai", model=self.model).observe(time.monotonic() - start)

    async def ainvoke_messages(self, messages: list) -> str:
        start = time.monotonic()
        try:
            response = await self._llm.ainvoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        finally:
            LLM_REQUESTS.labels(provider="openai", model=self.model).inc()
            LLM_LATENCY.labels(provider="openai", model=self.model).observe(time.monotonic() - start)


class AnthropicProvider:
    """LLM provider backed by the Anthropic Claude API."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic as _anthropic
        self.model = model
        self._client = _anthropic.AsyncAnthropic(api_key=api_key)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        start = time.monotonic()
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        finally:
            LLM_REQUESTS.labels(provider="anthropic", model=self.model).inc()
            LLM_LATENCY.labels(provider="anthropic", model=self.model).observe(time.monotonic() - start)

    async def ainvoke_messages(self, messages: list) -> str:
        # Convert LangChain message objects to Anthropic dicts.
        converted = []
        system_text = ""
        for msg in messages:
            if hasattr(msg, "type"):
                if msg.type == "system":
                    system_text = msg.content
                elif msg.type == "human":
                    converted.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    converted.append({"role": "assistant", "content": msg.content})
            else:
                converted.append(msg)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 256,
            "messages": converted or [{"role": "user", "content": str(messages)}],
        }
        if system_text:
            kwargs["system"] = system_text

        start = time.monotonic()
        try:
            response = await self._client.messages.create(**kwargs)
            return response.content[0].text if response.content else ""
        finally:
            LLM_REQUESTS.labels(provider="anthropic", model=self.model).inc()
            LLM_LATENCY.labels(provider="anthropic", model=self.model).observe(time.monotonic() - start)


def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider based on settings."""
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set in .env"
            )
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set in .env"
            )
        return OpenAIProvider(settings.openai_api_key, settings.openai_model)
    return OllamaProvider(settings.llm_model, settings.ollama_base_url)
