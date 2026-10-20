"""Unit tests for the section-aware, token-windowed chunker."""

from __future__ import annotations

from rag_platform.ingestion.chunker import _ENCODING, chunk_filing

RISK_FACTORS_TEXT = (
    "ITEM 1A. RISK FACTORS\n" + ("Our business faces significant competition risks. " * 400)
)
LEGAL_PROCEEDINGS_TEXT = (
    "ITEM 3. LEGAL PROCEEDINGS\n" + "We are involved in various litigation matters. " * 5
)
TWO_SECTION_FILING = RISK_FACTORS_TEXT + "\n" + LEGAL_PROCEEDINGS_TEXT


def test_chunks_never_straddle_sections() -> None:
    chunks = chunk_filing(TWO_SECTION_FILING, metadata={})

    risk_chunks = [c for c in chunks if c.metadata["item"] == "ITEM 1A"]
    legal_chunks = [c for c in chunks if c.metadata["item"] == "ITEM 3"]

    assert risk_chunks and legal_chunks
    assert all("LEGAL PROCEEDINGS" not in c.text for c in risk_chunks)
    assert all("RISK FACTORS" not in c.text for c in legal_chunks)


def test_chunks_respect_max_token_size() -> None:
    chunks = chunk_filing(RISK_FACTORS_TEXT, metadata={}, chunk_size_tokens=500, chunk_overlap_tokens=100)

    assert len(chunks) > 1
    assert all(len(_ENCODING.encode(c.text)) <= 500 for c in chunks)


def test_consecutive_chunks_share_overlap() -> None:
    chunks = chunk_filing(RISK_FACTORS_TEXT, metadata={}, chunk_size_tokens=500, chunk_overlap_tokens=100)

    first_tokens = _ENCODING.encode(chunks[0].text)
    second_tokens = _ENCODING.encode(chunks[1].text)

    overlap = set(first_tokens[-100:]) & set(second_tokens[:100])
    assert overlap


def test_short_section_returns_single_chunk() -> None:
    chunks = chunk_filing(LEGAL_PROCEEDINGS_TEXT, metadata={})

    assert len(chunks) == 1
    assert chunks[0].metadata["item"] == "ITEM 3"
    assert chunks[0].metadata["chunk_index"] == 0


def test_metadata_propagates_to_every_chunk() -> None:
    chunks = chunk_filing(TWO_SECTION_FILING, metadata={"ticker": "AAPL", "accession": "0000320193-24-000123"})

    assert chunks
    assert all(c.metadata["ticker"] == "AAPL" for c in chunks)
    assert all(c.metadata["accession"] == "0000320193-24-000123" for c in chunks)
