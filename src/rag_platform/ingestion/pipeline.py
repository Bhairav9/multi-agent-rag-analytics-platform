"""End-to-end ingestion CLI: EDGAR download -> chunk -> embed -> index."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from qdrant_client import QdrantClient

from rag_platform.config import get_settings
from rag_platform.ingestion.chunker import chunk_filing
from rag_platform.ingestion.sec_edgar import download_filing, list_filings, lookup_cik
from rag_platform.retrieval.bm25 import FastEmbedSparse
from rag_platform.retrieval.embeddings import FastEmbedDense
from rag_platform.retrieval.qdrant_store import QdrantStore


def run_pipeline(tickers: list[str], years: tuple[int, ...], raw_dir: Path) -> tuple[int, int]:
    settings = get_settings()
    store = QdrantStore(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection)
    store.ensure_collection()
    dense, sparse = FastEmbedDense(), FastEmbedSparse()

    n_filings = n_chunks = 0
    with httpx.Client(timeout=30.0) as client:
        for ticker in tickers:
            cik = lookup_cik(ticker, client)
            for filing in list_filings(ticker, cik, client, years=years):
                path = download_filing(filing, client, raw_dir)
                text = path.read_text(encoding="utf-8", errors="ignore")
                chunks = chunk_filing(
                    text,
                    metadata={
                        "ticker": ticker,
                        "item": None,
                        "form": filing.form,
                        "accession": filing.accession,
                        "source_url": str(path),
                    },
                    chunk_size_tokens=settings.chunk_size_tokens,
                    chunk_overlap_tokens=settings.chunk_overlap_tokens,
                )
                if not chunks:
                    continue
                ids = [
                    store.point_id(ticker, filing.accession, str(i)) for i in range(len(chunks))
                ]
                texts = [c.text for c in chunks]
                store.upsert(
                    ids=ids,
                    texts=texts,
                    payloads=[c.metadata for c in chunks],
                    dense_vectors=dense.embed(texts),
                    sparse_vectors=sparse.embed(texts),
                )
                n_filings += 1
                n_chunks += len(chunks)
    return n_filings, n_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC filings into Qdrant.")
    parser.add_argument("--tickers", required=True, help="Comma-separated, e.g. AAPL,MSFT")
    parser.add_argument("--years", required=True, help="Comma-separated fiscal years, e.g. 2024")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    years = tuple(int(y.strip()) for y in args.years.split(","))
    n_filings, n_chunks = run_pipeline(tickers, years, raw_dir=Path("data/raw"))
    print(f"Indexed {n_filings} filings, {n_chunks} chunks.")


if __name__ == "__main__":
    main()

