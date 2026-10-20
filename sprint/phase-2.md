# Phase 2 — LLM Provider

> Goal: end this phase with a working `OllamaProvider` that satisfies the `LLMProvider` Protocol — so any Phase 3 agent can call `await provider.astream(prompt)` and receive streaming tokens back without knowing or caring which model is running underneath. Unit tests are hermetic (no Ollama server, no network required). **No agents yet.** Just the LLM abstraction layer.

---

## Task checklist

| # | Task | File(s) | Status |
|---|---|---|---|
| 1 | `OllamaProvider` + `get_llm_provider()` factory | `src/rag_platform/llm/provider.py` | ⬜ |
| 2 | LLM provider unit tests (fake streaming) | `tests/unit/test_llm_provider.py` | ⬜ |

Phase 2 only touches `llm/`. `agents/`, `cache/`, `evaluation/`, `ui/` stay empty until their phases.

---

## Core concepts (read once before you type anything)

### 1. Why a Protocol, not a base class

Phase 0 introduced `LLMProvider` as a `typing.Protocol` stub:

```python
class LLMProvider(Protocol):
    model: str
    async def astream(self, prompt: str) -> AsyncIterator[str]: ...
```

A Protocol defines a **structural contract** — any class that has `model` and `astream` satisfies it automatically, with no inheritance required. This means:

- `OllamaProvider` satisfies it without inheriting from anything.
- `FakeLLMProvider` in tests satisfies it with 3 lines of code.
- Phase 3 agents declare `provider: LLMProvider` in their constructor and work with either.

Compare this to an ABC: an ABC forces every implementation to know about the base class. A Protocol lets you swap providers (Ollama → OpenAI → Groq) by changing one line in config, with zero changes to agent code.

*Used in:* Task 1, Task 2.

---

### 2. `AsyncIterator[str]` — what streaming actually means

`astream` returns an `AsyncIterator[str]`, not a `str`. This means the caller does:

```python
async for token in provider.astream("What is Apple's revenue?"):
    print(token, end="", flush=True)
```

Each `token` is a small piece of the response (often a word or sub-word). The LLM generates them one by one — so the first token appears in ~1s even if the full response takes 30s.

This is what enables the WebSocket streaming in Phase 4: tokens are forwarded to the browser as they arrive, giving a ChatGPT-like experience instead of a blank screen followed by a wall of text.

`langchain-ollama`'s `ChatOllama.astream()` already yields `AIMessageChunk` objects. Our wrapper extracts `.content` from each chunk and yields it as a plain `str`, so callers never touch LangChain types directly.

*Used in:* Task 1.

---

### 3. `ChatOllama` from `langchain-ollama`

`langchain-ollama` is already a runtime dependency (in `pyproject.toml`). It wraps the Ollama HTTP API and gives us:

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3.1:8b-instruct", base_url="http://ollama:11434")
async for chunk in llm.astream([HumanMessage(content="hello")]):
    print(chunk.content)
```

`OllamaProvider` is a thin wrapper around this — it translates a plain `str` prompt into a `HumanMessage`, calls `llm.astream`, and yields `chunk.content` as `str` tokens. That's all it does.

*Used in:* Task 1.

---

### 4. Factory function — `get_llm_provider()`

Rather than constructing `OllamaProvider` directly in agent code, a factory function reads config and returns the right provider:

```python
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.llm_model, settings.ollama_base_url)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
```

This is the same pattern as `get_settings()` from Phase 0 — one call, returns the configured object. Adding OpenAI support later means adding one `if` branch here, nothing else changes.

*Used in:* Task 1.

---

### 5. Hermetic tests — faking an async iterator

Testing streaming code is awkward if you need a real server. The fix: a `FakeLLMProvider` that returns a pre-defined list of tokens as an async iterator:

```python
class FakeLLMProvider:
    model = "fake-model"

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        for token in ["Hello", " world", "!"]:
            yield token
```

This satisfies `LLMProvider` Protocol, runs in-process with no network, and produces deterministic output — so assertions are simple:

```python
tokens = [t async for t in provider.astream("any prompt")]
assert tokens == ["Hello", " world", "!"]
```

For testing `OllamaProvider` specifically, we monkeypatch `ChatOllama` to return a fake async iterator instead of calling the real Ollama server.

*Used in:* Task 2.

---

## Order to type things

1. `llm/provider.py` — `OllamaProvider` class + `get_llm_provider()` factory.
2. `tests/unit/test_llm_provider.py` — hermetic tests with fake streaming.

---

## Task descriptions

### Task 1 — `src/rag_platform/llm/provider.py`

**What you're doing:** Replace the Phase 0 stub with the full implementation:
- Keep the existing `LLMProvider` Protocol (unchanged).
- Add `OllamaProvider` — wraps `ChatOllama`, translates `str → HumanMessage → AsyncIterator[str]`.
- Add `get_llm_provider()` — reads `settings.llm_provider` and returns the right provider.

**Why the project needs it:** Phase 3 agents call `get_llm_provider()` to get their LLM. Without this, every agent would construct `ChatOllama` directly — tying agent code to a specific backend, making tests require a real server, and making provider-swapping painful.

**Done when:** `get_llm_provider()` returns an `OllamaProvider` when `LLM_PROVIDER=ollama`, and `await provider.astream("hello")` yields at least one non-empty string token against a running Ollama server (verified manually, not in CI).

```python
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
```

---

### Task 2 — `tests/unit/test_llm_provider.py`

**What you're doing:** Three hermetic tests — no Ollama server, no network:
1. `OllamaProvider.astream` yields tokens from the underlying LLM.
2. `get_llm_provider()` returns `OllamaProvider` when `llm_provider="ollama"`.
3. `get_llm_provider()` raises `ValueError` for an unknown provider.

**Why the project needs it:** `OllamaProvider` is the only class in the codebase that calls an external service at runtime. Testing it with a fake removes the dependency on Ollama being up, keeps CI fast, and proves the token-extraction logic (`.content` stripping) is correct.

**Done when:** `uv run pytest tests/unit/test_llm_provider.py` passes with no network calls and no Ollama server running.

```python
"""Unit tests for LLM provider (hermetic — no Ollama server required)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_platform.llm.provider import OllamaProvider, get_llm_provider


class FakeChunk:
    def __init__(self, content: str) -> None:
        self.content = content


async def fake_astream(_messages):
    for token in ["Hello", " world", "!"]:
        yield FakeChunk(token)


def test_ollama_provider_streams_tokens() -> None:
    provider = OllamaProvider(model="llama3.1:8b-instruct", base_url="http://localhost:11434")

    with patch.object(provider._llm, "astream", side_effect=fake_astream):
        import asyncio
        tokens = asyncio.run(_collect(provider, "hi"))

    assert tokens == ["Hello", " world", "!"]


async def _collect(provider: OllamaProvider, prompt: str) -> list[str]:
    return [t async for t in provider.astream(prompt)]


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

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_provider()

    get_settings.cache_clear()
```

---

## Definition of done for Phase 2

- `get_llm_provider()` returns `OllamaProvider` when `LLM_PROVIDER=ollama` in `.env`.
- `OllamaProvider.astream(prompt)` yields `str` tokens (verified against running Ollama manually with `make run` + `docker compose up`).
- `uv run pytest` (hermetic) is green with no network calls and ≥80% coverage on `llm/provider.py`.
- `ruff check .`, `ruff format --check .`, `mypy src` all pass.
- Phase 3 agents can now do `provider = get_llm_provider()` and call `await provider.astream(prompt)`.
