# Architecture

## System Overview

```
Browser
  │
  ▼
Streamlit UI (port 8501)
  │  WebSocket  ws://app:8000/ws/chat?api_key=...
  ▼
FastAPI App (port 8000)
  │
  ├── Fast paths (no graph, instant response):
  │     greeting shortcut · off-topic filter · Redis cache hit
  │
  └── LangGraph pipeline:
        START
          │
        supervisor ──► needs_web=True  ──► web_search (DuckDuckGo)
          │                                      │
          └──► needs_web=False ──► retrieval ────┘
                                      │  hybrid RRF (Qdrant, port 6333)
                                      │  filtered by ticker payload
                                      ▼
                                  synthesizer (LLM stream)
                                      │  Claude / OpenAI / Ollama
                                      ▼
                                    critic ──► APPROVED ──► END
                                      │
                                      └──► REVISE ──► synthesizer (max 2 loops)

Redis (port 6380 host → 6379 container) — exact-match cache + conversation history
Prometheus (port 9090) — scrapes /metrics every 15 s
Grafana (port 3000)    — dashboards over Prometheus data
```

---

## Component Breakdown

### 1. Streamlit UI (`src/rag_platform/ui/streamlit_app.py`)
- Single-page chat interface
- Opens a persistent WebSocket connection per query
- Streams tokens in real time using `websockets` async client
- Reads `WS_URL` and `API_KEY` from environment

### 2. FastAPI App (`src/rag_platform/api/`)

| File | Responsibility |
|---|---|
| `main.py` | App factory, lifespan startup (Qdrant, Redis, graph init) |
| `routes.py` | `GET /healthz` (liveness) and `GET /healthz/ready` (readiness, auth-protected) |
| `websocket.py` | WebSocket chat handler, fast-path shortcuts, cache logic, graph streaming |
| `auth.py` | `X-API-Key` header dependency used on protected endpoints |
| `metrics.py` | Prometheus counters: `rag_cache_hits_total`, `rag_cache_misses_total`, `rag_llm_requests_total`, `rag_llm_latency_seconds` |

### 3. LangGraph Pipeline (`src/rag_platform/agents/`)

```
START → supervisor ──► retrieval ──► synthesizer ──► critic ──► END
                  └──► web_search ──┘              │
                                    ◄──────────────┘ (self-correction, max 2 revisions)
```

**supervisor node** (`supervisor.py`)
- Classifies the query: SEC corpus (local) vs. general web question
- Sets `needs_web` flag to route to `retrieval` or `web_search`

**retrieval node** (`retrieval.py`)
- Detects the company ticker from the query text (`_detect_ticker`)
- Calls `HybridRetriever.retrieve(query, ticker=ticker)` — Qdrant filters to that company's chunks only
- Returns `retrieved_chunks` list

**web_search node** (`web_search.py`)
- Runs a DuckDuckGo (default) or Tavily search for off-corpus questions
- Configured by `WEB_SEARCH_PROVIDER` env var; falls back to DuckDuckGo if `TAVILY_API_KEY` is missing
- Returns `web_results` list

**synthesizer node** (`synthesizer.py`)
- Builds a prompt from `retrieved_chunks` + `web_results` + `query` + `chat_history`
- Streams tokens through the configured LLM provider
- Returns `draft_answer`, `final_answer`, `citations`

**critic node** (`critic.py`)
- Fact-checks the draft answer against retrieved chunks (Qdrant) or web results (when query was web-routed)
- Returns `APPROVED` → sets `final_answer` and routes to `END`
- Returns `REVISE` → clears `final_answer`, increments `revision_count`, loops back to `synthesizer`
- Hard cap: `MAX_REVISIONS = 2`

### 4. Retrieval Layer (`src/rag_platform/retrieval/`)

| File | Responsibility |
|---|---|
| `embeddings.py` | Dense embedder — FastEmbed BGE-small-en-v1.5 (384-dim) |
| `bm25.py` | Sparse embedder — FastEmbed BM25 |
| `hybrid.py` | `HybridRetriever` — embeds query, calls `QdrantStore.search()`, returns `RetrievedChunk` list |
| `qdrant_store.py` | Qdrant client wrapper — RRF fusion search with optional ticker payload filter |

**Hybrid search flow:**
```
query
  ├── dense embed  ──► BGE-small vector (384-dim)
  └── sparse embed ──► BM25 sparse vector
        │
        ▼
  Qdrant query_points (prefetch_limit=20 per index, fuse with RRF)
        │  optionally filtered: payload.ticker == "AAPL"
        ▼
  top-K scored chunks (default K=10, tunable via RETRIEVAL_TOP_K)
```

### 5. LLM Providers (`src/rag_platform/llm/provider.py`)

Selected by `LLM_PROVIDER` env var:

| Provider | Class | Speed | Cost |
|---|---|---|---|
| `ollama` | `OllamaProvider` | Slow (CPU) | Free |
| `anthropic` | `AnthropicProvider` | Fast | Pay-per-use |
| `openai` | `OpenAIProvider` | Fast | Pay-per-use |

All providers implement the `LLMProvider` protocol: `model`, `astream(prompt)`, `ainvoke_messages(messages)`.

### 6. Cache Layer (`src/rag_platform/cache/`)

| File | Responsibility |
|---|---|
| `redis_client.py` | Async Redis connection wrapper |
| `llm_cache.py` | Exact-match query cache — SHA-256 key, 24 h TTL |
| `conversation.py` | Per-session chat history stored as Redis list — 1 h TTL per turn |

**Cache key:** `llm:cache:<sha256(query.strip().lower())>`

### 7. Ingestion Pipeline (`src/rag_platform/ingestion/`)

```
SEC EDGAR API
  │  list_filings(ticker, cik, years)
  ▼
download_filing() → raw HTML/text saved to data/raw/
  │
chunk_filing() → overlapping token chunks (500 tok, 100 tok overlap)
  │  metadata: ticker, form, accession, source_url
  ▼
FastEmbedDense.embed() + FastEmbedSparse.embed()
  │
QdrantStore.upsert() → deterministic point IDs (re-ingest = overwrite)
```

### 8. Observability

| Tool | Purpose | Port |
|---|---|---|
| Prometheus | Scrapes `/metrics` every 15 s | 9090 |
| Grafana | Dashboards over Prometheus data | 3000 |
| LangSmith | Optional LangGraph trace (set `LANGSMITH_API_KEY`) | — |
| JSON logs | Structured logging via `logging_config.py` | stdout |

---

## Data Flow: Full Query Lifecycle

```
1. User types query in Streamlit
2. WebSocket sends {"query": "...", "session_id": "..."}
3. Server checks: greeting? → instant reply
4. Server checks: off-topic? (no SEC keywords) → redirect message
5. Server checks: Redis cache? → serve cached answer
6. Load conversation history from Redis
7. LangGraph: supervisor node
   a. Classify query: SEC corpus vs. general web question
   b. Set needs_web flag
8. LangGraph: retrieval node (needs_web=False) OR web_search node (needs_web=True)
   Retrieval:
   a. Detect ticker in query
   b. Embed query (dense + sparse)
   c. Qdrant RRF search, filtered by ticker
   d. Return top-K chunks
   Web search:
   a. DuckDuckGo (or Tavily) search
   b. Return web_results
9. LangGraph: synthesizer node
   a. Build prompt (chunks/web_results + history + query)
   b. Stream tokens from LLM
   c. Return draft_answer + citations
10. LangGraph: critic node
    a. Fact-check draft against source chunks or web results
    b. APPROVED → final_answer; REVISE → loop back (max 2 revisions)
11. Stream tokens back over WebSocket as {"type":"token"} frames
12. Send {"type":"done"} with citations
13. Cache result in Redis (24 h TTL)
14. Append turn to conversation history (1 h TTL)
```

---

## Directory Structure

```
src/rag_platform/
├── agents/
│   ├── graph.py          # LangGraph graph builder (supervisor → retrieval → synthesizer → critic)
│   ├── supervisor.py     # LLM-based router — sets needs_web flag
│   ├── retrieval.py      # Retrieval node + ticker detection
│   ├── synthesizer.py    # Synthesizer node (LLM streaming)
│   ├── critic.py         # Fact-checker — APPROVED/REVISE with self-correction loop
│   ├── web_search.py     # DuckDuckGo fallback for off-corpus queries
│   └── state.py          # AgentState TypedDict
├── api/
│   ├── auth.py           # X-API-Key dependency
│   ├── metrics.py        # Prometheus custom metrics
│   ├── routes.py         # /healthz, /healthz/ready
│   └── websocket.py      # WebSocket chat handler
├── cache/
│   ├── conversation.py   # Chat history in Redis
│   ├── llm_cache.py      # Exact-match LLM cache
│   └── redis_client.py   # Async Redis client
├── ingestion/
│   ├── chunker.py        # Token-based text chunker
│   ├── pipeline.py       # CLI entry point
│   └── sec_edgar.py      # EDGAR downloader
├── llm/
│   └── provider.py       # OllamaProvider, AnthropicProvider, OpenAIProvider
├── retrieval/
│   ├── bm25.py           # Sparse embedder (FastEmbed BM25)
│   ├── embeddings.py     # Dense embedder (FastEmbed BGE-small)
│   ├── hybrid.py         # HybridRetriever facade
│   └── qdrant_store.py   # Qdrant client wrapper
├── ui/
│   └── streamlit_app.py  # Streamlit chat UI
├── config.py             # Pydantic settings (from .env)
├── logging_config.py     # JSON structured logging
└── main.py               # FastAPI app factory + lifespan
```

---

## Design Decisions

### Hybrid Retrieval (BM25 + Dense)
Dense-only embeddings (BGE-small) miss exact financial terms — ticker symbols (AAPL), specific figures ($391B), acronyms (EBITDA, GAAP). BM25 recovers these via exact-term matching. RRF fusion formula: `score = Σ 1/(k + rank_i)`, k=60. Result: +41.7% recall@10 over dense-only (see [EVALUATION.md](EVALUATION.md)).

### Why Qdrant
Native named sparse+dense vectors with `FusionQuery` + `Prefetch` — no client-side merging needed. FAISS: dense-only, no persistence. Chroma: no native sparse. Weaviate: heavier operational footprint. Qdrant runs in a single container with a stable REST+gRPC API.

### Section-Aware Chunking
SEC filings have legally mandated ITEM sections (Item 1=Business, Item 1A=Risk Factors, Item 7=MD&A, Item 8=Financials). Splitting on `_ITEM_HEADER_RE` preserves section boundaries so a chunk never mixes Risk Factors with Financial Statements. Token-based (tiktoken cl100k_base, 500 tok / 100 tok overlap) rather than character-based because embedding models have token budgets, not character budgets.

### LLMProvider as Protocol (not ABC)
Protocol-based duck typing (PEP 544): `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider` satisfy `LLMProvider` without inheriting from it. Switching provider = set `LLM_PROVIDER` env var. Each provider emits the same Prometheus metrics (`LLM_LATENCY`, `LLM_REQUESTS`), so dashboards work regardless of backend.

### Exact-Match Cache (not Semantic)
SEC financial data is precise — "Apple FY2024 revenue" and "AAPL fiscal 2024 total revenue" are different questions deserving independent answers. A semantic cache would create false-positive hits and return stale or wrong numbers. SHA-256 of `query.strip().lower()` guarantees cache correctness. Cache hit bypasses the full graph: ~0.06 s vs ~87 s cold.

### Deterministic Point IDs (uuid5)
`QdrantStore.point_id(*parts)` generates uuid5 namespace IDs from `(ticker, accession, chunk_index)`. Re-ingesting the same filing upserts rather than duplicates. Makes the ingestion pipeline idempotent — safe to re-run after a partial failure or a filing update.
