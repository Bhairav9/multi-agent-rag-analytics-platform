# Phase 1 — Ingestion + Hybrid Retrieval

> Goal: end this phase with `python -m rag_platform.ingestion.pipeline --tickers AAPL,MSFT --years 2024` pulling real 10-K/10-Q filings from SEC EDGAR, chunking them section-aware, embedding them (dense + sparse), and indexing them into Qdrant — plus a `HybridRetriever.query()` that returns ranked, cited chunks. Unit tests for the chunker and the retrieval layer are hermetic (no network, no Qdrant/Docker required). **No agents, no LLM calls yet.** Just the data pipeline and the search index.

---

## Task checklist

| # | Task | File(s) | Status |
|---|---|---|---|
| 1 | Document ingestion/retrieval env vars | `.env.example` | ✅ |
| 2 | Extend typed config | `src/rag_platform/config.py` | ✅ |
| 3 | SEC EDGAR downloader | `src/rag_platform/ingestion/sec_edgar.py` | ✅ |
| 4 | Section-aware chunker | `src/rag_platform/ingestion/chunker.py` | ✅ |
| 5 | Dense embedder wrapper (fastembed / bge-small) | `src/rag_platform/retrieval/embeddings.py` | ✅ |
| 6 | Sparse embedder wrapper (fastembed / BM25) | `src/rag_platform/retrieval/bm25.py` | ✅ |
| 7 | Qdrant store (dense + sparse named vectors) | `src/rag_platform/retrieval/qdrant_store.py` | ✅ |
| 8 | Hybrid retriever (RRF fusion) | `src/rag_platform/retrieval/hybrid.py` | ⬜ |
| 9 | Ingestion pipeline CLI | `src/rag_platform/ingestion/pipeline.py` | ⬜ |
| 10 | Wire real `make ingest` target | `Makefile` | ⬜ |
| 11 | Chunker unit tests | `tests/unit/test_chunker.py` | ✅ |
| 12 | Qdrant store unit tests (in-memory) | `tests/unit/test_qdrant_store.py` | ✅ |
| 13 | Hybrid retriever unit tests (fake embedders) | `tests/unit/test_hybrid.py` | ⬜ |
| 14 | Golden-path integration test (opt-in, real fastembed) | `tests/integration/test_ingestion_pipeline.py` | ⬜ |
| 15 | `data/raw/`, `data/processed/` placeholders | `data/raw/.gitkeep`, `data/processed/.gitkeep` | ⬜ |

Phase 1 only touches `ingestion/` and `retrieval/`. `agents/`, `cache/`, `evaluation/`, `ui/` stay empty until their phases.

---

## Core concepts (read once before you type anything)

### 1. SEC EDGAR is a free API, but it has rules

EDGAR's `data.sec.gov` endpoints are public and keyless, but SEC **requires every request to send an identifying `User-Agent` header** (`"Your Name your.email@example.com"`) and asks you to stay under ~10 requests/second. Skip the header and you get `403 Forbidden`. This isn't optional — it's the one "auth" mechanism EDGAR has.

Two endpoints matter:
- `https://www.sec.gov/files/company_tickers.json` — ticker → CIK (Central Index Key) lookup table.
- `https://data.sec.gov/submissions/CIK{10-digit-zero-padded}.json` — every filing a company has ever made, with form type (`10-K`, `10-Q`), accession number, and primary document filename.

From the accession number + filename you construct the actual filing URL: `https://www.sec.gov/Archives/edgar/data/{cik}/{accession-no-dashes}/{primary-doc}`.

*Used in:* Task 3.

---

### 2. Section-aware chunking (why not just "split every 500 words")

A 10-K has a rigid structure — `Item 1. Business`, `Item 1A. Risk Factors`, `Item 7. MD&A`, etc. Naive fixed-size chunking will happily cut a chunk in half across an item boundary, mixing "Risk Factors" text with "Legal Proceedings" text in one chunk. That destroys retrieval precision — a query about risk factors can match a chunk that's mostly about litigation.

**Our approach (two passes):**
1. **Split on item headers first** using a regex (`ITEM\s+\d+[A-Z]?\.`), so each section is chunked independently and carries its section name as metadata (`Document(page_content=..., metadata={"item": "1A", "ticker": "AAPL", ...})`).
2. **Then apply a sliding token window** (500 tokens, 100-token overlap) *within* each section, so oversized sections (Risk Factors is often 20+ pages) still produce retrievable chunks, and each chunk retains enough surrounding context via the overlap.

**Why the overlap?** Without it, a sentence split exactly at the 500-token boundary loses its second half in one chunk and its first half in the next — neither chunk is a complete, retrievable unit.

*Used in:* Task 4.

---

### 3. Dense vs. sparse retrieval — what each is good at

- **Dense (embedding cosine similarity)** — good at *semantic* matches: "profitability concerns" retrieves a chunk that says "margin pressure" even with zero shared words.
- **Sparse (BM25 — term-frequency statistics)** — good at *exact* matches: ticker symbols, dollar figures, product names, defined legal terms. Financial text is full of these, and dense embeddings tend to blur them (all numbers "look similar" in embedding space).

Neither wins alone on financial text — that's the whole premise of the resume's "+22% Recall@10 via hybrid" claim. Phase 1 builds the machinery; Phase 5 proves the number.

*Used in:* Tasks 5, 6, 7, 8.

---

### 4. Qdrant named vectors — one collection, two vector spaces

Instead of running two separate databases (a vector DB + Elasticsearch for BM25), Qdrant lets a single **point** (one chunk) carry **multiple named vectors**:

```python
client.upsert(
    collection_name="sec_filings",
    points=[
        PointStruct(
            id=chunk_id,
            vector={"dense": [0.12, -0.03, ...], "sparse": SparseVector(indices=[...], values=[...])},
            payload={"text": "...", "ticker": "AAPL", "item": "1A", "source_url": "..."},
        )
    ],
)
```

One collection, one point per chunk, two ways to search it. This is *why* the stack picked Qdrant over "Qdrant + a second BM25 engine."

*Used in:* Task 7.

---

### 5. `fastembed` — local, keyless embeddings for both vector types

`fastembed` (from Qdrant) runs ONNX models locally, no API key, no network at query time (after the model is downloaded once):
- `TextEmbedding(model_name="BAAI/bge-small-en-v1.5")` → dense vectors.
- `SparseTextEmbedding(model_name="Qdrant/bm25")` → sparse vectors (indices + values), computed with real BM25 statistics, not a neural approximation.

Both ship as thin wrapper classes in `retrieval/embeddings.py` and `retrieval/bm25.py` rather than calling `fastembed` directly from `qdrant_store.py` — see Concept 8 below for why.

*Used in:* Tasks 5, 6.

---

### 6. Reciprocal Rank Fusion (RRF) — combining two rankings without tuning weights

Given a dense ranking and a sparse ranking for the same query, RRF scores each document by:

$$\text{score}(d) = \sum_{\text{rankings } r} \frac{1}{k + \text{rank}_r(d)}$$

with $k = 60$ conventionally. Documents that rank well in *either* list score well overall; documents that rank well in *both* dominate. No weight-tuning between dense and sparse scores (which live on incomparable scales — cosine similarity vs. BM25 score) — RRF only cares about **rank position**, so it sidesteps the scale-mismatch problem entirely.

Qdrant runs this **server-side**:

```python
client.query_points(
    collection_name="sec_filings",
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=20),
        Prefetch(query=sparse_vector, using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
)
```

*Used in:* Task 8.

---

### 7. Structural typing again — `Embedder` Protocols for hermetic tests

Phase 0 introduced `typing.Protocol` for `LLMProvider`. Phase 1 reuses the same trick for embedders:

```python
class DenseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SparseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVector]: ...
```

`FastEmbedDense` / `FastEmbedSparse` (the real ONNX-backed classes) satisfy these Protocols. So do `FakeDenseEmbedder` / `FakeSparseEmbedder` test doubles that return fixed, deterministic vectors with zero model download and zero disk I/O.

**Why this matters for the org rule "hermetic unit tests, no network":** downloading a real fastembed ONNX model (~100 MB) on a cold CI runner is a real network call and takes real time — it does not belong in a unit test that must run in milliseconds. `HybridRetriever` and `QdrantStore` are built to depend on the *Protocol*, not the concrete fastembed class, so unit tests inject fakes and only the opt-in integration test (Task 14) touches the real models.

*Used in:* Tasks 5, 6, 8, 13.

---

### 8. Qdrant's in-memory mode — hermetic vector-store tests

`qdrant-client` can run **fully in-process** with no server:

```python
from qdrant_client import QdrantClient

client = QdrantClient(location=":memory:")
```

This gives `test_qdrant_store.py` a real Qdrant engine (real collection creation, real named vectors, real RRF fusion) with **zero Docker, zero network, zero external state** — the same hermeticity guarantee `TestClient` gave the FastAPI tests in Phase 0.

*Used in:* Task 12.

---

### 9. Idempotent upsert — deterministic point IDs

If `make ingest` is re-run (new filing added, or you just re-ran it by habit), it must not create duplicate chunks. The fix: derive each point's ID deterministically from its content, e.g. `uuid5(NAMESPACE, f"{ticker}:{accession}:{item}:{chunk_index}")`. Same input → same ID → Qdrant's `upsert` *overwrites* the existing point instead of adding a new one. This is why `pipeline.py` never needs a "check if already indexed" step — it's naturally idempotent.

*Used in:* Tasks 7, 9.

---

### 10. `python -m package.module` CLI pattern

`ingestion/pipeline.py` gets an `if __name__ == "__main__":` block with `argparse`, so it's runnable both as `python -m rag_platform.ingestion.pipeline --tickers AAPL,MSFT --years 2024` (matches §10 NFR "one-command ingest") and importable (`from rag_platform.ingestion.pipeline import run_pipeline`) for tests and the future `make ingest` wrapper. Same reasoning as the FastAPI app factory in Phase 0 — logic lives in a testable function; the `if __name__` block is just a thin CLI shell around it.

*Used in:* Task 9.

---

## Order to type things

1. `.env.example` + `config.py` → new settings exist before anything reads them.
2. `ingestion/sec_edgar.py` → can fetch and save raw filings standalone.
3. `ingestion/chunker.py` → pure function, test it in isolation first (no network needed).
4. `retrieval/embeddings.py` + `retrieval/bm25.py` → embedder wrappers + Protocols.
5. `retrieval/qdrant_store.py` → collection creation + upsert, tested with `:memory:`.
6. `retrieval/hybrid.py` → RRF query facade, tested with fake embedders.
7. `ingestion/pipeline.py` → wire steps 2–6 together behind a CLI.
8. `Makefile` → real `ingest` target.
9. Unit tests throughout (write alongside each module, not just at the end — TDD per org rules).
10. Integration test last, marked opt-in.

---

## Task descriptions (what you're actually doing, and why)

### Task 1 — Extend `.env.example`

**What you're doing:** Adding every new env var the ingestion/retrieval layer reads: `SEC_EDGAR_USER_AGENT`, `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `RETRIEVAL_TOP_K`.

**Why the project needs it:** Same 12-factor rule as Phase 0 — nothing hardcoded, everything documented in one place.

**Done when:** Every new `Settings` field from Task 2 has a matching line here.

```dotenv
# --- Ingestion (SEC EDGAR) ---
SEC_EDGAR_USER_AGENT=Your Name your.email@example.com   # required by SEC, no default

# --- Chunking ---
CHUNK_SIZE_TOKENS=500
CHUNK_OVERLAP_TOKENS=100

# --- Retrieval ---
RETRIEVAL_TOP_K=10
```

---

### Task 2 — Extend `src/rag_platform/config.py`

**What you're doing:** Adding the four fields above to the existing `Settings` class.

**Concept to understand:** Same fail-fast boundary-validation pattern from Phase 0. `sec_edgar_user_agent` has no safe default (SEC will reject requests without a real contact string), so it's required — leaving it blank in `.env` should be caught by ingestion code at the start of `sec_edgar.py`, not silently sent as an empty header.

**Done when:** `get_settings().chunk_size_tokens == 500` and the new fields round-trip through `.env`.

```python
# added to Settings, alongside existing fields
sec_edgar_user_agent: str = ""
chunk_size_tokens: int = 500
chunk_overlap_tokens: int = 100
retrieval_top_k: int = 10
```

---

### Task 3 — `src/rag_platform/ingestion/sec_edgar.py`

**What you're doing:** A small client with three functions: `lookup_cik(ticker) -> str`, `list_filings(cik, forms=("10-K", "10-Q")) -> list[FilingMeta]`, and `download_filing(filing: FilingMeta) -> Path` (saves raw HTML/text under `data/raw/{ticker}/{accession}.html`).

**Why the project needs it:** This is the only network-touching part of ingestion. Isolating it in one module means the chunker and everything downstream can be tested without ever hitting `sec.gov`.

**Concept to understand:** Every request must set `headers={"User-Agent": settings.sec_edgar_user_agent}`. CIKs from `company_tickers.json` are ints; the submissions endpoint needs them zero-padded to 10 digits (`f"{cik:010d}"`).

**Done when:** `download_filing(...)` for a real ticker saves a non-empty file under `data/raw/`, and unit tests cover `lookup_cik`/`list_filings` parsing logic against a **saved sample JSON fixture** (not a live call) using `respx` to mock `httpx`.

```python
"""SEC EDGAR downloader — ticker lookup, filing listing, raw document fetch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from rag_platform.config import get_settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"


@dataclass(frozen=True)
class FilingMeta:
    ticker: str
    cik: str
    form: str
    accession: str
    primary_document: str
    filing_date: str


def _headers() -> dict[str, str]:
    ua = get_settings().sec_edgar_user_agent
    if not ua:
        raise RuntimeError("SEC_EDGAR_USER_AGENT must be set before calling SEC EDGAR.")
    return {"User-Agent": ua}


def lookup_cik(ticker: str, client: httpx.Client) -> str:
    """Resolve a ticker symbol to a zero-padded 10-digit CIK."""
    resp = client.get(TICKERS_URL, headers=_headers())
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return f"{entry['cik_str']:010d}"
    raise ValueError(f"Unknown ticker: {ticker}")


def list_filings(
    ticker: str,
    cik: str,
    client: httpx.Client,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    years: tuple[int, ...] = (),
) -> list[FilingMeta]:
    """Return filing metadata for a CIK, filtered by form type and fiscal year."""
    resp = client.get(SUBMISSIONS_URL.format(cik=cik), headers=_headers())
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    filings = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        filing_date = recent["filingDate"][i]
        if years and int(filing_date[:4]) not in years:
            continue
        filings.append(
            FilingMeta(
                ticker=ticker,
                cik=cik,
                form=form,
                accession=recent["accessionNumber"][i],
                primary_document=recent["primaryDocument"][i],
                filing_date=filing_date,
            )
        )
    return filings


def download_filing(filing: FilingMeta, client: httpx.Client, raw_dir: Path) -> Path:
    """Download one filing's primary document to data/raw/{ticker}/{accession}.html."""
    accession_nodash = filing.accession.replace("-", "")
    url = ARCHIVES_URL.format(
        cik=int(filing.cik), accession_nodash=accession_nodash, doc=filing.primary_document
    )
    resp = client.get(url, headers=_headers())
    resp.raise_for_status()

    out_dir = raw_dir / filing.ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filing.accession}.html"
    out_path.write_bytes(resp.content)
    return out_path
```

---

### Task 4 — `src/rag_platform/ingestion/chunker.py`

**What you're doing:** A pure function `chunk_filing(text: str, metadata: dict) -> list[Chunk]` that (1) splits on `ITEM \d+[A-Z]?\.` headers, then (2) sliding-windows each section into overlapping token chunks.

**Why the project needs it:** This is the single highest-leverage file for the "+22% recall" claim — bad chunking caps retrieval quality no matter how good the embeddings are.

**Concept to understand:** Token counting via `tiktoken` (`cl100k_base` encoding is a reasonable proxy even though we're not using an OpenAI model at inference time — it's just a consistent, fast token counter). No network calls, no model loading.

**Done when:** Feeding it a fixture 10-K excerpt string returns chunks that (a) never straddle two `Item` sections, (b) are each ≤500 tokens, (c) share ~100 tokens with their neighbor.

```python
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
    sections = _ITEM_HEADER_RE.split(text)
    return [s.strip() for s in sections if s.strip()]


def _sliding_window(tokens: list[int], size: int, overlap: int) -> list[list[int]]:
    if len(tokens) <= size:
        return [tokens]
    step = size - overlap
    return [tokens[i : i + size] for i in range(0, len(tokens), step) if tokens[i : i + size]]


def chunk_filing(
    text: str,
    metadata: dict,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 100,
) -> list[Chunk]:
    """Split filing text into section-aware, overlapping token windows."""
    chunks: list[Chunk] = []
    for section_idx, section in enumerate(_split_sections(text)):
        tokens = _ENCODING.encode(section)
        for window in _sliding_window(tokens, chunk_size_tokens, chunk_overlap_tokens):
            chunk_text = _ENCODING.decode(window)
            chunks.append(
                Chunk(text=chunk_text, metadata={**metadata, "section_index": section_idx})
            )
    return chunks
```

---

### Task 5 — `src/rag_platform/retrieval/embeddings.py`

**What you're doing:** A `DenseEmbedder` Protocol + a `FastEmbedDense` implementation wrapping `fastembed.TextEmbedding`.

**Why the project needs it:** Isolates the only place that loads the `bge-small` ONNX model, so every other module (and every unit test) depends on the Protocol, not this concrete class.

**Done when:** `FastEmbedDense().embed(["hello"])` returns a `list[list[float]]` of length 384 (bge-small's dimension). Not exercised in unit tests — only in the Task 14 integration test.

```python
"""Dense embedding wrapper (fastembed / BAAI/bge-small-en-v1.5)."""

from __future__ import annotations

from typing import Protocol

from rag_platform.config import get_settings


class DenseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedDense:
    """Dense embedder backed by a local fastembed ONNX model. No API key, no network after first download."""

    def __init__(self, model_name: str | None = None) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name or get_settings().embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]
```

---

### Task 6 — `src/rag_platform/retrieval/bm25.py`

**What you're doing:** The sparse-side mirror of Task 5 — `SparseEmbedder` Protocol + `FastEmbedSparse` wrapping `fastembed.SparseTextEmbedding` (`Qdrant/bm25`).

**Done when:** `FastEmbedSparse().embed(["hello"])` returns objects with `.indices` and `.values` (Qdrant's `SparseVector` shape).

```python
"""Sparse embedding wrapper (fastembed / Qdrant/bm25)."""

from __future__ import annotations

from typing import Protocol

from qdrant_client.models import SparseVector


class SparseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVector]: ...


class FastEmbedSparse:
    """BM25 sparse embedder backed by fastembed. No API key, no network after first download."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=vec.indices.tolist(), values=vec.values.tolist())
            for vec in self._model.embed(texts)
        ]
```

---

### Task 7 — `src/rag_platform/retrieval/qdrant_store.py`

**What you're doing:** A `QdrantStore` class: `ensure_collection()` (creates `dense` + `sparse` named vectors if missing), `upsert(chunks, dense_vectors, sparse_vectors)` (deterministic point IDs per Concept 9), and a thin `search(...)` used by `hybrid.py`.

**Why the project needs it:** Single place that knows Qdrant's point/collection schema — nobody else constructs a `PointStruct` by hand.

**Concept to understand:** Constructor takes a `QdrantClient` instance rather than building one internally from `settings.qdrant_url` — this is **dependency injection**, and it's what lets Task 12's tests pass in an in-memory client instead of a real server.

**Done when:** Against an in-memory client, `ensure_collection()` + `upsert(...)` + `search(...)` round-trips a known chunk back out.

```python
"""Qdrant collection management — dense + sparse named vectors, one point per chunk."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

_ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def ensure_collection(self, dense_dim: int = 384) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    @staticmethod
    def point_id(*parts: str) -> str:
        """Deterministic point ID so re-ingesting the same chunk overwrites, never duplicates."""
        return str(uuid.uuid5(_ID_NAMESPACE, ":".join(parts)))

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        payloads: list[dict],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
    ) -> None:
        points = [
            PointStruct(
                id=ids[i],
                vector={"dense": dense_vectors[i], "sparse": sparse_vectors[i]},
                payload={**payloads[i], "text": texts[i]},
            )
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        top_k: int = 10,
        prefetch_limit: int = 20,
    ) -> list[dict]:
        result = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=prefetch_limit),
                Prefetch(query=sparse_vector, using="sparse", limit=prefetch_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [{"score": p.score, **p.payload} for p in result.points]
```

---

### Task 8 — `src/rag_platform/retrieval/hybrid.py`

**What you're doing:** `HybridRetriever` — the public-facing retrieval API. Takes a `DenseEmbedder`, a `SparseEmbedder`, and a `QdrantStore` (all injected), and exposes `retrieve(query: str, top_k: int) -> list[RetrievedChunk]`.

**Why the project needs it:** This is the one class the future Retrieval Agent (Phase 3) will call. Keeping embedding + fusion logic behind this facade means the agent code never touches Qdrant or fastembed directly.

**Done when:** `retriever.retrieve("risk factors")` returns chunks sorted by fused RRF score, each carrying its citation metadata (`ticker`, `item`, `source_url`).

```python
"""Public hybrid retrieval facade — embeds a query and runs RRF fusion search."""

from __future__ import annotations

from dataclasses import dataclass

from rag_platform.retrieval.bm25 import SparseEmbedder
from rag_platform.retrieval.embeddings import DenseEmbedder
from rag_platform.retrieval.qdrant_store import QdrantStore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


class HybridRetriever:
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        store: QdrantStore,
    ) -> None:
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        dense_vector = self._dense.embed([query])[0]
        sparse_vector = self._sparse.embed([query])[0]
        hits = self._store.search(dense_vector, sparse_vector, top_k=top_k)
        return [
            RetrievedChunk(
                text=hit.pop("text"),
                score=hit.pop("score"),
                metadata=hit,
            )
            for hit in hits
        ]
```

---

### Task 9 — `src/rag_platform/ingestion/pipeline.py`

**What you're doing:** The CLI orchestrator: parse `--tickers`, `--years`; for each ticker, look up CIK, list filings, download, chunk, embed (dense + sparse), and upsert into Qdrant.

**Why the project needs it:** This is the "one-command ingest" from discuss.md §10, and the thing that turns Tasks 3–8 from isolated modules into an actual working pipeline.

**Done when:** `python -m rag_platform.ingestion.pipeline --tickers AAPL --years 2024` populates the Qdrant collection and prints a summary (`N filings, M chunks indexed`).

```python
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
```

---

### Task 10 — Wire real `make ingest`

**What you're doing:** Replacing the Phase 0 echo stub with the real command, accepting `TICKERS`/`YEARS` make variables.

**Done when:** `make ingest TICKERS=AAPL,MSFT YEARS=2024` runs the pipeline.

```makefile
ingest:
	uv run python -m rag_platform.ingestion.pipeline --tickers $(TICKERS) --years $(YEARS)
```

---

### Task 11 — `tests/unit/test_chunker.py`

**What you're doing:** Hermetic tests against `chunk_filing` with small synthetic strings — no filing download, no tokenizer network call (tiktoken's `cl100k_base` ships/caches locally after first use, same caveat as fastembed; the CI runner should warm this cache once, or the test suite accepts one cached download).

**Done when:** Tests assert: section boundaries are respected, chunk token counts are bounded, and metadata propagates onto every chunk.

```python
"""Unit tests for the section-aware chunker."""

from rag_platform.ingestion.chunker import chunk_filing


def test_splits_on_item_headers() -> None:
    text = "ITEM 1. BUSINESS\nWe make phones.\nITEM 1A. RISK FACTORS\nSupply chain risk."
    chunks = chunk_filing(text, metadata={"ticker": "AAPL"})
    assert len(chunks) == 2
    assert "BUSINESS" in chunks[0].text
    assert "RISK FACTORS" in chunks[1].text


def test_respects_chunk_size_and_metadata() -> None:
    text = "ITEM 1. BUSINESS\n" + ("word " * 2000)
    chunks = chunk_filing(text, metadata={"ticker": "MSFT"}, chunk_size_tokens=500)
    assert all(len(c.text.split()) <= 550 for c in chunks)  # small margin for the header
    assert all(c.metadata["ticker"] == "MSFT" for c in chunks)
```

---

### Task 12 — `tests/unit/test_qdrant_store.py`

**What you're doing:** Tests using `QdrantClient(location=":memory:")` — a real Qdrant engine, zero network.

```python
"""Unit tests for QdrantStore using an in-memory Qdrant engine."""

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag_platform.retrieval.qdrant_store import QdrantStore


def test_upsert_and_search_roundtrip() -> None:
    store = QdrantStore(QdrantClient(location=":memory:"), "test_collection")
    store.ensure_collection(dense_dim=4)

    store.upsert(
        ids=[store.point_id("AAPL", "acc-1", "0")],
        texts=["Apple's risk factors include supply chain disruption."],
        payloads=[{"ticker": "AAPL"}],
        dense_vectors=[[0.1, 0.2, 0.3, 0.4]],
        sparse_vectors=[SparseVector(indices=[1, 5], values=[0.9, 0.4])],
    )

    hits = store.search(
        dense_vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=SparseVector(indices=[1, 5], values=[0.9, 0.4]),
        top_k=1,
    )
    assert hits[0]["ticker"] == "AAPL"
```

---

### Task 13 — `tests/unit/test_hybrid.py`

**What you're doing:** Tests `HybridRetriever` with fake dense/sparse embedders (deterministic fixed vectors) and the in-memory `QdrantStore` — no real fastembed models loaded.

```python
"""Unit tests for HybridRetriever using fake embedders + in-memory Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag_platform.retrieval.hybrid import HybridRetriever
from rag_platform.retrieval.qdrant_store import QdrantStore


class FakeDense:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeSparse:
    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 5], values=[0.9, 0.4]) for _ in texts]


def test_retrieve_returns_cited_chunks() -> None:
    store = QdrantStore(QdrantClient(location=":memory:"), "test_collection")
    store.ensure_collection(dense_dim=4)
    store.upsert(
        ids=[store.point_id("AAPL", "acc-1", "0")],
        texts=["Apple's risk factors include supply chain disruption."],
        payloads=[{"ticker": "AAPL", "item": "1A"}],
        dense_vectors=[[0.1, 0.2, 0.3, 0.4]],
        sparse_vectors=[SparseVector(indices=[1, 5], values=[0.9, 0.4])],
    )

    retriever = HybridRetriever(FakeDense(), FakeSparse(), store)
    results = retriever.retrieve("supply chain risk", top_k=1)

    assert results[0].metadata["ticker"] == "AAPL"
    assert "supply chain" in results[0].text.lower()
```

---

### Task 14 — `tests/integration/test_ingestion_pipeline.py` (opt-in)

**What you're doing:** One end-to-end test using the **real** `FastEmbedDense`/`FastEmbedSparse` models (downloaded once, then cached) against a small saved sample filing fixture — no live SEC EDGAR call, but real embedding models. Marked so it's excluded from the default hermetic `pytest` run.

**Why the project needs it:** Unit tests with fakes prove the *wiring* is correct; this test proves the *real* fastembed vectors actually round-trip through Qdrant with sane results. It belongs in `tests/integration/` (already an empty package) precisely because it's slower and touches real model weights.

**Concept to understand:** `@pytest.mark.integration` + a `pyproject.toml` addition (`markers = ["integration: slow, downloads real models"]`) and excluding it by default via `-m "not integration"` in CI's fast job, or running it as a separate, non-blocking CI step.

**Done when:** `uv run pytest -m integration` passes locally (first run downloads ~100 MB of models, subsequent runs are fast from cache).

---

### Task 15 — `data/raw/.gitkeep`, `data/processed/.gitkeep`

**What you're doing:** Empty placeholder files so the folder shape survives in Git even though `.gitignore` (Phase 0, Task 1) excludes their contents.

**Done when:** `git status` after running `make ingest` shows new files under `data/raw/AAPL/...` as *ignored*, not untracked-and-visible.

---

## Config additions summary (`pyproject.toml`)

No new runtime dependencies — `httpx`, `qdrant-client`, `fastembed`, `tiktoken` are not yet all pinned; confirm `tiktoken` is added (it wasn't in the Phase 0 dependency list) and `respx` (already a dev dep) covers the EDGAR HTTP mocking in unit tests.

```toml
dependencies = [
    # ...Phase 0 deps unchanged...
    "tiktoken>=0.8.0",
]

[tool.pytest.ini_options]
markers = [
    "integration: slower tests that download real ML models (excluded by default)",
]
addopts = "-ra --strict-markers -m \"not integration\" --cov=rag_platform --cov-report=term-missing --cov-report=xml"
```

---

## Definition of done for Phase 1

- `make ingest TICKERS=AAPL YEARS=2024` downloads at least one real 10-K, chunks it, and indexes it into a running Qdrant.
- `HybridRetriever.retrieve("...")` returns non-empty, cited results against that index.
- `uv run pytest` (default, hermetic) is green with no network calls and ≥80% coverage on the new modules.
- `uv run pytest -m integration` (opt-in) passes once real models are cached.
- `ruff check .`, `ruff format --check .`, `mypy src` all pass.
