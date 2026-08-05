"""LLM provider abstraction. Filled in Phase 2."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class LLMProvider(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    model: str

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """Yield response tokens as they are generated."""
        ...