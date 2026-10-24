"""End-to-end ingestion CLI: EDGAR download -> chunk -> embed -> index."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
from qdrant_client import QdrantClient

from rag_platform.config import get_settings
from rag_platform.ingestion.chunker import chunk_filing
from rag_platform.ingestion.sec_edgar import download_filing, list_filings, lookup_cik
from rag_platform.retrieval.bm25 import FastEmbedSparse
from rag_platform.retrieval.embeddings import FastEmbedDense
from rag_platform.retrieval.qdrant_store import QdrantStore


def _make_store(settings) -> QdrantStore:
    """Create a fresh QdrantStore with a new connection (avoids idle-timeout on cloud NAT)."""
    return QdrantStore(
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=120),
        settings.qdrant_collection,
    )


def run_pipeline(tickers: list[str], years: tuple[int, ...], raw_dir: Path) -> tuple[int, int]:
    settings = get_settings()
    # Ensure collection + payload index exist using a short-lived connection.
    _make_store(settings).ensure_collection()
    dense, sparse = FastEmbedDense(), FastEmbedSparse()

    n_filings = n_chunks = 0
    with httpx.Client(timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=30.0)) as client:
        for ticker in tickers:
            print(f"[{ticker}] Looking up CIK...", flush=True)
            cik = lookup_cik(ticker, client)
            print(f"[{ticker}] CIK={cik}. Listing filings...", flush=True)
            filings = list(list_filings(ticker, cik, client, years=years))
            print(f"[{ticker}] {len(filings)} filings found.", flush=True)
            for filing in filings:
                print(f"[{ticker}] Downloading {filing.form} {filing.accession}...", flush=True)
                path = download_filing(filing, client, raw_dir)
                text = path.read_text(encoding="utf-8", errors="ignore")
                print(f"[{ticker}] Chunking {len(text):,} chars...", flush=True)
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
                    print(f"[{ticker}] No chunks — skipping.", flush=True)
                    continue
                ids = [
                    QdrantStore.point_id(ticker, filing.accession, str(i))
                    for i in range(len(chunks))
                ]
                texts = [c.text for c in chunks]
                print(f"[{ticker}] Embedding {len(chunks)} chunks...", flush=True)
                dense_vecs = dense.embed(texts)
                sparse_vecs = sparse.embed(texts)
                print(f"[{ticker}] Upserting to Qdrant...", flush=True)
                # Fresh connection per attempt: avoids stale-TCP and transient DNS failures.
                for upsert_attempt in range(5):
                    try:
                        store = _make_store(settings)
                        store.upsert(
                            ids=ids,
                            texts=texts,
                            payloads=[c.metadata for c in chunks],
                            dense_vectors=dense_vecs,
                            sparse_vectors=sparse_vecs,
                        )
                        break
                    except Exception as exc:
                        if upsert_attempt >= 4:
                            raise
                        wait = 15 * (2 ** upsert_attempt)
                        print(f"  Upsert attempt {upsert_attempt + 1}/5 failed: {exc}. Retrying in {wait}s...", flush=True)
                        time.sleep(wait)
                n_filings += 1
                n_chunks += len(chunks)
                print(f"  [{ticker}] {filing.form} {filing.accession}: {len(chunks)} chunks (total {n_chunks})", flush=True)
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

