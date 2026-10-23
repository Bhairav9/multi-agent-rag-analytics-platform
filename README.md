# Multi-Agent RAG Analytics Platform

A portfolio-grade, production-ready system for answering natural-language questions about SEC financial filings (10-K, 10-Q) using a multi-agent LangGraph pipeline, hybrid vector retrieval, and a Streamlit chat UI.

## Architecture

```
User → Streamlit UI → FastAPI WebSocket
                           │
                    LangGraph Graph
                    ┌──────┴──────┐
                Supervisor      (decides web vs. local)
                    │
          ┌─────────┴──────────┐
       Retrieval           Web Search
    (Qdrant hybrid)       (DuckDuckGo)
          │
       Synthesizer  ←── context (chunks + web)
          │
        Critic  ──► self-correction loop (max 2 revisions)
          │
      final_answer
```

**Key components:**

| Layer | Technology |
|---|---|
| LLM | Ollama (local) · Anthropic Claude · OpenAI GPT |
| Embeddings | FastEmbed — BGE-small (dense) + BM25 (sparse) |
| Vector store | Qdrant (hybrid RRF search) |
| Cache | Redis (exact-match SHA-256, 24 h TTL) |
| Orchestration | LangGraph (StateGraph with conditional routing) |
| API | FastAPI + WebSocket streaming |
| UI | Streamlit |
| Evaluation | TruLens (groundedness, relevance, coherence) |
| Observability | LangSmith tracing (optional) |

## Performance

Benchmarked on 20 hand-labeled ground-truth questions across AAPL, MSFT, GOOGL, TSLA, JPM (see [EVALUATION.md](EVALUATION.md) for full methodology and reproduction steps).

| Metric | Value |
|---|---|
| Retrieval Recall@10 — hybrid RRF | **0.85** |
| Retrieval Recall@10 — dense-only | 0.60 |
| Hybrid improvement over dense | **+41.7%** |
| Cold pipeline P50 (full graph, CPU) | 87.2 s |
| Cache hit P50 (Redis exact-match) | 0.063 s |
| Cache speedup | **1384x** |

> Cold path: supervisor → retrieval (Qdrant RRF) → synthesizer (LLM stream) → critic (self-correction, max 2 revisions).
> Warm path: Redis SHA-256 exact-match cache hit — graph is bypassed entirely.

## Prerequisites

- [Podman](https://podman.io/) (or Docker) with Compose
- Python 3.11+ with [uv](https://docs.astral.sh/uv/)

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd 1-Multi-Agent\ RAG\ Analytics\ Platform

# 2. Copy and configure environment
cp .env.example .env
# Edit .env — set SEC_EDGAR_USER_AGENT to "FirstName LastName email@example.com"

# 3. Start all services (Qdrant, Redis, Ollama, app, Streamlit)
make up

# 4. Pull the LLM model (one-time, ~2 GB)
make pull-models

# 5. Ingest SEC filings (AAPL, MSFT, GOOGL, TSLA, JPM — FY 2022-2024)
make ingest TICKERS="AAPL MSFT GOOGL TSLA JPM" YEARS="2022 2023 2024"

# 6. Open the chat UI
open http://localhost:8501   # macOS
# or navigate to http://<podman-vm-ip>:8501 on Windows/WSL2
```

## Cloud Deployment (Railway)

The app runs on Railway with Anthropic as the LLM backend — no GPU or Ollama needed.

**Prerequisites (all free tier):**
- [Railway account](https://railway.app) — connect your GitHub repo
- [Qdrant Cloud](https://cloud.qdrant.io) — free 1 GB cluster
- [Upstash](https://upstash.com) — free Redis (use the `rediss://` TLS URL)
- [Anthropic API key](https://console.anthropic.com)

**Steps:**
1. Push this repo to GitHub
2. Create a new Railway project → "Deploy from GitHub repo"
3. Set these environment variables in the Railway dashboard:

| Variable | Value |
|---|---|
| `LLM_PROVIDER` | `anthropic` |
| `ANTHROPIC_API_KEY` | your key |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` |
| `QDRANT_URL` | your Qdrant Cloud URL |
| `REDIS_URL` | your Upstash `rediss://` URL |
| `API_KEY` | any secret string |
| `SEC_EDGAR_USER_AGENT` | `"FirstName LastName email@example.com"` |

4. Railway builds from `Dockerfile` and deploys automatically (config in `railway.toml`).
5. The FastAPI app is live at `https://<your-app>.up.railway.app`
6. The Streamlit UI can be deployed as a second Railway service pointing `WS_URL` at the app URL.

> **Note:** Ollama, Prometheus, and Grafana are not deployed to Railway — they are local-only services defined in `docker-compose.yml`.

## Services & Ports

| Service | Port | Purpose |
|---|---|---|
| FastAPI app | 8000 | REST API + WebSocket |
| Streamlit UI | 8501 | Chat interface |
| Qdrant | 6333 | Vector store |
| Redis | 6380 | Cache + conversation history |
| Ollama | 11434 | Local LLM inference |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `anthropic` \| `openai` |
| `LLM_MODEL` | `llama3.1:8b-instruct` | Ollama model name (Ollama mode only) |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama service URL |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Claude model to use |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `API_KEY` | `change-me` | Shared secret for HTTP + WebSocket auth |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant service URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `RETRIEVAL_TOP_K` | `10` | Chunks returned per query (use `3` on CPU) |
| `CACHE_TTL_SECONDS` | `86400` | Redis cache TTL (24 h) |
| `SEC_EDGAR_USER_AGENT` | — | **Required.** `"Name email"` for SEC EDGAR API |
| `LANGSMITH_API_KEY` | — | Optional. Enables LangSmith tracing |

## Cloud Deployment (OpenAI mode)

Running with `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` removes the need for a local GPU/Ollama. This makes the app trivially deployable to Railway, Render, or any container host.

### Managed cloud services (free tiers)

| Component | Local (Docker) | Cloud (free tier) |
|---|---|---|
| Vector store | Qdrant container | [Qdrant Cloud](https://cloud.qdrant.io) — 1 GB free |
| Cache | Redis container | [Redis Cloud](https://redis.com/try-free/) — 30 MB free |
| LLM | Ollama (local) | OpenAI API or Anthropic API |

### Deploy to Railway

```bash
# 1. Install Railway CLI
npm install -g @railway/cli && railway login

# 2. Create a new project
railway init

# 3. Link to this repo and set environment variables in the Railway dashboard:
#    LLM_PROVIDER=openai
#    OPENAI_API_KEY=sk-...
#    OPENAI_MODEL=gpt-4o-mini
#    QDRANT_URL=https://<cluster>.qdrant.io:6333
#    QDRANT_API_KEY=<qdrant-cloud-key>        # add to qdrant_store.py if needed
#    REDIS_URL=redis://:<password>@<host>:port/0
#    API_KEY=<strong-random-secret>
#    SEC_EDGAR_USER_AGENT=FirstName LastName email@example.com

# 4. Deploy
railway up
```

### Deploy to Render

1. Fork this repo and connect it to [Render](https://render.com).
2. Create two **Web Services** — one for the FastAPI app, one for Streamlit.
3. For the FastAPI service set:
   - **Build command**: `pip install -e .`
   - **Start command**: `uvicorn rag_platform.main:app --host 0.0.0.0 --port $PORT`
4. For the Streamlit service set:
   - **Start command**: `streamlit run src/rag_platform/ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Environment variable**: `WS_URL=wss://<fastapi-service>.onrender.com/ws/chat`
5. Set all required env vars (see table above) in each service's dashboard.

### docker-compose without Ollama (cloud LLM mode)

Comment out or remove the `ollama` service and its `depends_on` block from `docker-compose.yml`:

```yaml
# Remove these lines when using LLM_PROVIDER=openai or anthropic:
#  ollama:
#    image: ollama/ollama:0.4.4
#    ...
```

Then point to your cloud Qdrant and Redis:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export QDRANT_URL=https://<cluster>.qdrant.io:6333
export REDIS_URL=redis://:<pw>@<host>:port/0
docker compose up app streamlit prometheus grafana
```

## Development

```bash
# Install dependencies
make install

# Run tests with coverage (≥80% threshold)
make test

# Lint + format
make lint
make format

# Type check
make typecheck

# Run FastAPI locally (without Docker)
make run

# Run Streamlit UI locally
make ui
```

## Health Checks

```bash
# Liveness (process is up)
curl http://localhost:8000/healthz

# Readiness (Redis + Qdrant reachable)
curl http://localhost:8000/healthz/ready
```

## Project Structure

```
src/rag_platform/
├── agents/          # LangGraph nodes: supervisor, retrieval, synthesizer, critic
├── api/             # FastAPI routes + WebSocket endpoint
├── cache/           # Redis LLM cache + conversation history
├── config.py        # Pydantic settings (from .env)
├── ingestion/       # SEC EDGAR downloader + chunking pipeline
├── llm/             # LLMProvider Protocol + OllamaProvider
├── retrieval/       # Dense/sparse embedders, Qdrant store, HybridRetriever
├── ui/              # Streamlit chat application
└── evaluation/      # TruLens groundedness eval + latency/retrieval benchmarks

tests/
├── conftest.py      # Sets TESTING=true before imports
└── unit/            # Hermetic unit tests (fake LLM, fake retriever, fake Redis)
```

## Phases

| Phase | Description |
|---|---|
| 0 | Project skeleton, Docker Compose, CI |
| 1–3 | Hybrid retrieval (RRF), LLM provider abstraction, multi-agent LangGraph graph |
| 4 | Redis cache (exact-match + conversation history) |
| 5 | TruLens evaluation + latency/retrieval benchmarks |
| 6 | Streamlit chat UI with real-time WebSocket streaming |

## Performance Notes

See the [Performance](#performance) section above for benchmarked numbers. Additional notes:

- On CPU, `llama3.2:3b` produces ~7 tok/s; `llama3.1:8b` ~2–5 tok/s.
- Set `RETRIEVAL_TOP_K=3` when running on CPU to stay within the 2048-token context window.
- Use `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=openai` to eliminate GPU dependency entirely.

## License

MIT
