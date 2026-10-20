"""Section-aware, token-windowed chunker for SEC filings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

_ITEM_HEADER_RE = re.compile(r"(?=^\s*ITEM\s+\d+[A-Z]?\.)", re.IGNORECASE | re.MULTILINE)
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _split_sections(text: str) -> list[str]:
    """Split filing text into sections, one per ITEM header."""
    sections = _ITEM_HEADER_RE.split(text)
    return [s.strip() for s in sections if s.strip()]


def _sliding_window(tokens: list[int], size: int, overlap: int) -> list[list[int]]:
    """Slide a token window of `size` tokens with `overlap` tokens shared between windows."""
    if len(tokens) <= size:
        return [tokens]

    step = size - overlap

    windows = []
    for i in range(0, len(tokens), step):
        window = tokens[i : i + size]
        if window:
            windows.append(window)
    return windows


def chunk_filing(
    text: str,
    metadata: dict,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 100,
) -> list[Chunk]:
    """Split filing text into section-aware, overlapping token windows."""
    chunks: list[Chunk] = []

    for section_idx, section in enumerate(_split_sections(text)):
        item_match = re.match(r"\s*(ITEM\s+\d+[A-Z]?)\.", section, re.IGNORECASE)
        item_label = item_match.group(1).upper() if item_match else None

        tokens = _ENCODING.encode(section)
        windows = _sliding_window(tokens, chunk_size_tokens, chunk_overlap_tokens)

        for chunk_idx, window in enumerate(windows):
            chunk_text = _ENCODING.decode(window)
            chunk_metadata = {
                **metadata,
                "item": item_label,
                "section_index": section_idx,
                "chunk_index": chunk_idx,
            }
            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))

    return chunks
