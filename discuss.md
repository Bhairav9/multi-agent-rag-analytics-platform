# Enterprise Multi-Agent RAG Analytics Platform — Design Discussion

> Goal: Build a **portfolio-grade, resume-defensible** project that a recruiter or hiring manager at a product-based company (FAANG-adjacent, fintech, AI startups) can open on GitHub and immediately understand the depth of engineering. Every claim on the resume must be **reproducible** from the repo (code + metrics + docs).

---

## 1. Positioning & Recruiter Lens

### What the resume claims (and what we must be able to prove)
| Resume Claim | Proof we must ship in the repo |
|---|---|
| Multi-agent RAG using LangGraph (Retrieval, Web-Search, Critic) | Working LangGraph state machine with 3+ named agent nodes + diagram |
| +22% retrieval recall via hybrid (Qdrant dense + BM25 sparse) | Evaluation script + benchmark table (dense-only vs hybrid) on a fixed test set |
| −35% LLM latency via Redis multi-turn caching + async WebSocket streaming | Benchmark script (cached vs uncached) + FastAPI WebSocket demo |
| ≥94% groundedness via TruLens RAG Triad monitoring | TruLens dashboard screenshots + exported metrics JSON in `/evaluation/` |

**Rule:** if we can’t reproduce a number with a script in the repo, we don’t put it on the resume.

### What makes a GenAI project stand out on GitHub
1. **README that a non-ML PM can skim in 60 seconds** — architecture diagram, GIF demo, one-line quickstart.
2. **Reproducible evaluation** — `make eval` prints the exact numbers on the resume.
3. **Clean domain choice** — financial 10-K / earnings reports (SEC EDGAR is free, large, and “enterprise-y”).
4. **Production hygiene** — Docker, env-based config, structured logging, tests, CI.
5. **Observability** — TruLens + Prometheus/Grafana or at least LangSmith traces.
6. **Not another toy chatbot** — has agents, tools, evaluation, streaming, caching.

---

## 2. Domain Choice — Financial Documents

We’ll build the RAG corpus over **SEC 10-K / 10-Q / earnings-call transcripts**.

Why:
- Free, public, huge (SEC EDGAR API).
- “Enterprise financial data” matches the resume verbatim.
- Documents are long, structured (Item 1A Risk Factors, Item 7 MD&A), perfect for chunking + hybrid retrieval.
- Recruiters immediately recognize “financial analytics” as high-value.

Corpus plan:
- 20–30 companies (AAPL, MSFT, GOOGL, TSLA, JPM, GS, etc.), last 3 fiscal years.
- ~200–500 documents, ~50–150 MB text.
- Small enough to run locally, large enough to be non-trivial.

Example user queries the system must answer:
- *“Compare Apple’s and Microsoft’s FY2024 risk factors related to AI regulation.”*
- *“What did Tesla say about margin pressure in the last two earnings calls? Cite pages.”*
- *“Summarize JPMorgan’s exposure to commercial real estate as of the latest 10-K, and cross-check with recent news.”*

The third query is the killer demo — it **requires** all three agents (Retrieval + Web-Search + Critic).

---

## 3. High-Level Architecture

```
                  ┌────────────────────────┐
                  │  React / Streamlit UI  │
                  │  (chat + streaming)    │
                  └───────────┬────────────┘
                              │ WebSocket
                  ┌───────────▼────────────┐
                  │      FastAPI Gateway    │
                  │  (auth, WS, REST)       │
                  └───────────┬────────────┘
                              │
              ┌───────────────▼────────────────┐
              │      LangGraph Orchestrator     │
              │  ┌───────────────────────────┐  │
              │  │  Supervisor / Planner     │  │
              │  └─────┬──────────┬──────────┘  │
              │        │          │             │
              │  ┌─────▼───┐ ┌────▼─────┐  ┌───────────┐
              │  │Retrieval│ │Web-Search│  │  Critic   │
              │  │ Agent   │ │  Agent   │  │  Agent    │
              │  └─────┬───┘ └────┬─────┘  └─────┬─────┘
              └────────┼──────────┼──────────────┼─────┘
                       │          │              │
              ┌────────▼───┐ ┌────▼────┐   ┌────▼────┐
              │  Qdrant    │ │ Tavily/ │   │ TruLens │
              │ (dense +   │ │  Serper │   │  RAG    │
              │  BM25)     │ │  API    │   │  Triad  │
              └────────┬───┘ └─────────┘   └─────────┘
                       │
              ┌────────▼──────┐
              │     Redis     │
              │ (conv cache + │
              │  session mem) │
              └───────────────┘
```

### Agent responsibilities
- **Supervisor (LangGraph entry node)**: reads the user query + chat history, decides which agents to call and in what order. Implemented as a router node with structured output (Pydantic).
- **Retrieval Agent**: performs hybrid search (dense + BM25 + RRF fusion) over Qdrant, applies MMR re-ranking, returns top-k chunks with citations.
- **Web-Search Agent**: only invoked when the Supervisor decides the internal corpus is insufficient (e.g., “latest news”, “recent event”). Uses Tavily or Serper API. Sanitizes and chunks results.
- **Critic Agent**: reads the draft answer + retrieved context, checks for (a) unsupported claims, (b) missing citations, (c) contradictions. Can send the query back to Retrieval with a refined sub-query (self-correction loop, max 2 iterations).
- **Synthesizer** (part of Supervisor or a dedicated node): produces the final grounded answer with inline citations `[doc_id, page]`.

---

## 4. Tech Stack (final, opinionated)

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Matches resume, first-class state machine, good visualizer |
| LLM | **OpenAI GPT-4o-mini** (default) + **Groq Llama-3.1-70B** fallback | Cheap, fast, easy to swap; show provider-agnostic design |
| Embeddings | **OpenAI text-embedding-3-small** or **BAAI/bge-small-en-v1.5** (local) | Small model = free local option for recruiters cloning the repo |
| Vector DB | **Qdrant** (Docker) | Matches resume; supports dense + sparse (BM25) natively via `SparseVector` |
| Sparse retrieval | **Qdrant BM25** (fastembed `Qdrant/bm25`) | Same store, no separate ES cluster needed |
| Fusion | **Reciprocal Rank Fusion (RRF)** | Standard, easy to explain |
| Cache / Memory | **Redis** (Docker) | Matches resume; used for both LLM response cache and conversation history |
| API | **FastAPI** + **WebSockets** | Async streaming, matches resume |
| Frontend | **Streamlit** (v1) → **Next.js** (v2, optional) | Streamlit ships fast; Next.js if we want to look really polished |
| Evaluation | **TruLens** (RAG Triad: context relevance, groundedness, answer relevance) | Matches resume verbatim |
| Tracing | **LangSmith** (optional, free tier) | Nice screenshots for the README |
| Web search | **Tavily API** (500 free searches/mo) | Purpose-built for LLM agents |
| Doc ingestion | **unstructured** or **PyMuPDF** + custom SEC parser | 10-Ks are semi-structured |
| Config | **pydantic-settings** + `.env` | 12-factor |
| Packaging | **uv** or **poetry** | Fast, modern |
| Containerization | **Docker Compose** (app + qdrant + redis) | One-command spin-up |
| CI | **GitHub Actions** — lint (ruff), type (mypy), tests (pytest), coverage ≥80% | Enforces org standards |
| Tests | **pytest** + **pytest-asyncio** + **respx** (HTTP mocks) | Hermetic |

---

## 5. Repository Structure (proposed)

```
1-Multi-Agent RAG Analytics Platform/
├── README.md                    # Hero doc: diagram, GIF, quickstart, metrics table
├── ARCHITECTURE.md              # Deep dive: agent graph, state schema, decisions
├── EVALUATION.md                # How we produced the 22% / 35% / 94% numbers
├── docker-compose.yml           # app + qdrant + redis
├── Dockerfile
├── pyproject.toml
├── .env.example
├── Makefile                     # make ingest / make run / make eval / make test
├── .github/workflows/ci.yml
│
├── src/
│   └── rag_platform/
│       ├── __init__.py
│       ├── config.py            # pydantic-settings
│       ├── main.py              # FastAPI entrypoint
│       │
│       ├── api/
│       │   ├── routes.py        # REST endpoints
│       │   └── websocket.py     # /ws/chat streaming
│       │
│       ├── agents/
│       │   ├── graph.py         # LangGraph build_graph()
│       │   ├── state.py         # AgentState (TypedDict)
│       │   ├── supervisor.py
│       │   ├── retrieval.py
│       │   ├── web_search.py
│       │   ├── critic.py
│       │   └── synthesizer.py
│       │
│       ├── retrieval/
│       │   ├── qdrant_store.py  # dense + sparse collections
│       │   ├── hybrid.py        # RRF fusion, MMR
│       │   ├── bm25.py
│       │   └── embeddings.py
│       │
│       ├── cache/
│       │   ├── redis_client.py
│       │   ├── llm_cache.py     # semantic + exact key cache
│       │   └── conversation.py  # multi-turn history
│       │
│       ├── ingestion/
│       │   ├── sec_edgar.py     # download 10-K / 10-Q
│       │   ├── chunker.py       # section-aware chunking
│       │   └── pipeline.py      # end-to-end index build
│       │
│       ├── evaluation/
│       │   ├── trulens_setup.py
│       │   ├── retrieval_bench.py   # dense vs hybrid recall@k
│       │   ├── latency_bench.py     # cached vs uncached p50/p95
│       │   └── datasets/            # golden Q&A set (JSONL)
│       │
│       └── ui/
│           └── streamlit_app.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── docs/
│   ├── images/
│   │   ├── architecture.png
│   │   └── demo.gif
│   └── adr/                     # architecture decision records
│
└── data/
    ├── raw/                     # gitignored
    └── processed/               # gitignored
```

---

## 6. LangGraph State Schema (draft)

```python
class AgentState(TypedDict):
    # inputs
    query: str
    chat_history: list[BaseMessage]
    session_id: str

    # planning
    plan: list[str]            # ordered list of agent names to invoke
    needs_web: bool

    # retrieval
    retrieved_docs: list[Document]
    web_results: list[Document]

    # generation
    draft_answer: str
    final_answer: str
    citations: list[Citation]

    # critic loop
    critic_feedback: str | None
    revision_count: int        # capped at 2

    # observability
    trace_id: str
    metrics: dict              # latencies, cache hits, tokens
```

Edges:
- `START → supervisor`
- `supervisor → retrieval` (always)
- `supervisor → web_search` (conditional on `needs_web`)
- `retrieval / web_search → synthesizer`
- `synthesizer → critic`
- `critic → synthesizer` (if `revision_count < 2` and feedback exists)
- `critic → END` otherwise

---

## 7. Hybrid Retrieval — How we hit +22% recall

**Setup for the benchmark:**
1. Build a golden set: 100 questions with human-labeled relevant chunk IDs from our SEC corpus. (Can bootstrap with GPT-4o generating Q&A pairs from known chunks, then hand-verify 100.)
2. Metrics: **Recall@5**, **Recall@10**, **MRR@10**, **nDCG@10**.
3. Baselines:
   - (A) Dense-only (Qdrant, cosine, top-k=10)
   - (B) BM25-only
   - (C) Hybrid (dense + BM25 with RRF, k=60, then top-10)
   - (D) Hybrid + MMR re-rank
4. Publish the table in `EVALUATION.md`. Expect hybrid to win by 15–30% on Recall@10 vs dense-only — 22% is realistic for financial data (lots of tickers, numeric IDs, exact phrases that dense embeddings blur).

**Key implementation details:**
- Use Qdrant’s **named vectors** to store dense + sparse in the same point.
- Use `fastembed` for BM25 sparse vectors (`Qdrant/bm25`).
- Query with `prefetch=[dense_prefetch, sparse_prefetch]` and `FusionQuery(fusion=Fusion.RRF)` — server-side RRF.
- Chunking: section-aware (split on 10-K item headers) with 500-token windows and 100-token overlap.

---

## 8. Redis Caching + WebSocket Streaming — How we hit −35% latency

Two orthogonal wins compound:

**A. Multi-turn conversational cache (Redis)**
- **Exact-match cache**: hash of `(normalized_query, retrieved_doc_ids, model)` → cached response. TTL 24h.
- **Semantic cache** (optional stretch): embed the query, KNN search over a small Qdrant collection of previous queries; if cosine > 0.95, return cached answer.
- **Conversation memory**: last N turns stored in Redis under `session:{id}:history` (with TTL) so repeated follow-ups reuse prior retrieval results (retrieval cache) — this is where the “multi-turn” part of the resume claim lives.

**B. Async token streaming (FastAPI WebSocket)**
- Server-sent tokens via `async for chunk in llm.astream(...)` piped to `websocket.send_text(chunk)`.
- Perceived latency drops dramatically vs waiting for the full response — measure **TTFT (time to first token)**, not just end-to-end.

**Benchmark script:**
- Run 50 realistic queries twice (cold + warm).
- Report: end-to-end p50/p95, TTFT p50/p95, cache hit rate.
- Expect: 30–50% improvement on warm runs → 35% is defensible.

---

## 9. TruLens Evaluation — How we hit ≥94% groundedness

RAG Triad:
1. **Context Relevance** — is the retrieved context relevant to the query?
2. **Groundedness** — is every claim in the answer supported by the retrieved context?
3. **Answer Relevance** — does the answer address the query?

Setup:
- Wrap the LangGraph app with `TruChain` / custom `TruCustomApp`.
- Feedback functions via `LiteLLM` provider (OpenAI as judge).
- Run nightly (or on `make eval`) over a 50-question evaluation set.
- Export JSON metrics + screenshots of the TruLens dashboard into `docs/images/`.
- To defend ≥94% groundedness: (a) strong Critic agent forcing citations, (b) synthesizer prompted to refuse when context is insufficient, (c) hybrid retrieval improving context quality.

---

## 10. Non-Functional Requirements (portfolio quality bar)

- **One-command bring-up**: `docker compose up` → app on `localhost:8000`, UI on `localhost:8501`.
- **One-command ingest**: `make ingest TICKERS=AAPL,MSFT YEARS=2022,2023,2024`.
- **One-command evaluation**: `make eval` → prints the resume numbers.
- **Test coverage ≥80%** on changed code (matches org instructions).
- **Structured logging** (JSON, `structlog`) with `trace_id` propagation.
- **Config via env** (`.env.example` documents every var).
- **CI**: ruff + mypy + pytest + coverage gate.
- **README with GIF demo** — recorded via `asciinema` or a screen-capture GIF.

---

## 11. Roadmap (phased)

**Phase 0 — Skeleton (Day 1)**
- Repo scaffolding, `pyproject.toml`, docker-compose, `.env.example`, empty FastAPI app, health check, CI green.

**Phase 1 — Ingestion + Retrieval (Days 2–4)**
- SEC EDGAR downloader, chunker, Qdrant dense + sparse indexing.
- CLI: `python -m rag_platform.ingestion.pipeline`.
- Unit tests for chunker and retriever.

**Phase 2 — Single-agent RAG (Day 5)**
- Simple LangChain chain end-to-end, non-streaming. Baseline for latency + quality.

**Phase 3 — Multi-agent LangGraph (Days 6–8)**
- Supervisor + Retrieval + Web-Search + Critic + Synthesizer nodes.
- State schema + conditional edges + critic self-correction loop.

**Phase 4 — Streaming + Redis cache (Days 9–10)**
- WebSocket endpoint, token streaming, Redis exact-match cache, conversation memory.

**Phase 5 — Evaluation (Days 11–12)**
- Golden set, retrieval benchmark, latency benchmark, TruLens RAG Triad.
- Populate `EVALUATION.md` with real numbers.

**Phase 6 — UI + Polish (Days 13–14)**
- Streamlit chat UI with streaming, citation display, session picker.
- README hero: architecture diagram, GIF demo, metrics table, quickstart.

**Phase 7 (Stretch)**
- Semantic cache with Qdrant.
- Guardrails (PII redaction, prompt-injection detection).
- Next.js frontend.
- Deploy to Fly.io / Render with a public demo link.
- Add a `/metrics` endpoint (Prometheus) + Grafana dashboard.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| OpenAI cost during dev | Use `gpt-4o-mini` + Redis cache + `bge-small` local embeddings option |
| Recruiter can’t run it (no API keys) | Ship a **read-only demo mode** with pre-computed responses + hosted demo link |
| Numbers on resume don’t match repo | Every number produced by a script in `evaluation/`; CI re-runs a subset |
| “Just another RAG project” criticism | The **Critic self-correction loop** + **hybrid retrieval benchmark** + **TruLens dashboard** are the differentiators; highlight in README |
| Scope creep | Freeze Phases 0–6 as v1.0, tag it, then work on stretch as v1.1 |

---

## 13. What the README top-of-file should look like

```
# Enterprise Multi-Agent RAG Analytics Platform

Autonomous multi-agent RAG over SEC financial filings — LangGraph orchestrating
Retrieval, Web-Search, and Critic agents with hybrid Qdrant search, Redis-backed
multi-turn caching, FastAPI WebSocket streaming, and TruLens RAG-Triad evaluation.

[architecture diagram]
[demo GIF]

## Results (reproducible via `make eval`)
| Metric               | Baseline | This system | Δ      |
| Recall@10            | 0.63     | 0.77        | +22%   |
| End-to-end latency p50 | 4.8s   | 3.1s        | −35%   |
| Groundedness (TruLens)|  —      | 0.94        | —      |

## Quickstart
docker compose up
make ingest TICKERS=AAPL,MSFT
open http://localhost:8501
```

That single screen sells the project.

---

## 14. Open questions for you before we start coding

1. **LLM provider**: OpenAI only, or do you want Groq / local Ollama as a fallback path so recruiters without an OpenAI key can still run it?
2. **Frontend**: start with Streamlit (fast, ugly-ish) or invest in Next.js from the start (slower, hire-worthy polish)?
3. **Deployment**: are we shipping a live public demo link (Fly.io / Render / HuggingFace Spaces), or is a great local `docker compose up` enough?
4. **Data scope**: 20 tickers × 3 years feels right — agree, or do you want to start smaller (5 × 1 year) for faster iteration?
5. **Timeline**: aggressive 2-week build vs. relaxed 4-week? Impacts how much of Phase 7 we do.

Once you answer these five, we lock the plan and start on Phase 0 (repo skeleton + docker-compose + CI).

---

## 15. Locked Decisions (2026-07-30)

| # | Question | Decision |
|---|---|---|
| 1 | LLM provider | **OpenAI `gpt-4o-mini` (default) + local Ollama fallback** (`llama3.1:8b-instruct`). Embeddings: `text-embedding-3-small` **or** `bge-small-en-v1.5` via `fastembed` (no key needed). Selected via `LLM_PROVIDER` / `EMBEDDING_PROVIDER` env vars. |
| 2 | Frontend | **Streamlit only** (v1 = final). Next.js dropped from scope entirely. |
| 3 | Deployment | **Local `docker compose up` only.** No hosted demo. Invest saved time in a polished README GIF + a `DEMO_MODE=true` canned-response mode so key-less recruiters still see a working chat. |
| 4 | Data scope | **5 tickers × 1 year for v1** — `AAPL, MSFT, GOOGL, TSLA, JPM` × FY2024 10-K + latest 10-Q + latest earnings call (~15–20 docs). Ingestion pipeline stays generic (`--tickers`, `--years`) so scaling to 20×3 is a CLI flag. |
| 5 | Timeline | **Relaxed 4 weeks.** Phase 7 trimmed to: semantic cache (Qdrant), Prometheus `/metrics` + Grafana dashboard, guardrails (PII + prompt-injection). Next.js and live deploy explicitly **out**. |

### Deltas applied to earlier sections

- **§4 Tech Stack**
  - Add `LiteLLM` (or thin `LLMProvider` interface wrapping `langchain_openai` + `langchain_ollama`).
  - Add `ollama` service to `docker-compose.yml` behind a Compose profile (`--profile local-llm`) so it's opt-in and doesn't slow default startup.
  - Use `fastembed` for **both** BM25 sparse and local dense embeddings — single dependency, no key.
- **§5 Repo structure**
  - `src/rag_platform/llm/provider.py` — provider abstraction.
  - `src/rag_platform/demo/fixtures.jsonl` — canned Q&A for `DEMO_MODE`.
- **§10 NFRs** — `.env.example` documents `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `OLLAMA_BASE_URL`, `DEMO_MODE`.
- **§7 & §9 Eval set sizes** — 50 questions for retrieval bench, 30 for TruLens (right-sized for the 5×1 corpus; still statistically meaningful).
- **§11 Roadmap**
  - Insert **Phase 6.5 (Days ~18–20)**: record demo GIF, build `DEMO_MODE` fixture playback path through the WebSocket.
  - **Phase 7 (Days ~21–28)**: semantic cache → Prometheus/Grafana → guardrails, in that order; stop when runway ends.

### Ready-to-execute: Phase 0 checklist

- `pyproject.toml` (uv) with pinned deps: `fastapi`, `uvicorn[standard]`, `langgraph`, `langchain`, `langchain-openai`, `langchain-ollama`, `qdrant-client`, `fastembed`, `redis`, `pydantic-settings`, `structlog`, `streamlit`, `trulens-eval`, `tavily-python`.
- `docker-compose.yml`: `app`, `qdrant`, `redis`; `ollama` under profile `local-llm`.
- `.env.example` with every var documented.
- `src/rag_platform/{__init__,config,main}.py` — FastAPI app with `/healthz`.
- `Makefile` targets: `run`, `test`, `lint`, `ingest`, `eval` (stubs okay for now).
- `.github/workflows/ci.yml`: ruff + mypy + pytest + coverage gate (≥80% on changed files).
- `tests/unit/test_health.py` — first hermetic test, gets CI green.

Say "go" and I'll scaffold Phase 0.

---

## 16. Revision (2026-07-30, later): Zero paid services

**Constraint added:** no paid APIs, no paid tiers, no signup-required freemium keys. Everything must run offline on the developer's hardware with only free/OSS components.

### Component swaps

| Concern | Previous locked choice | New choice (fully free) |
|---|---|---|
| LLM (generator) | OpenAI `gpt-4o-mini` default | **Ollama `llama3.1:8b-instruct`** default. Low-RAM fallback: `qwen2.5:7b-instruct` or `llama3.2:3b`. |
| LLM (TruLens judge) | OpenAI as judge | **Ollama `qwen2.5:14b-instruct`** if hardware allows, else `llama3.1:8b`. Paired with a **rule-based NLI check** (`cross-encoder/nli-deberta-v3-base` via `sentence-transformers`) for a deterministic second opinion on groundedness. |
| Embeddings | `text-embedding-3-small` (OpenAI) | **`BAAI/bge-small-en-v1.5`** via `fastembed` (was already the fallback — now the default). |
| Web search | Tavily API (500/mo free tier — needs signup) | **`duckduckgo-search`** (no key, no signup). Tavily kept behind `WEB_SEARCH_PROVIDER=tavily` if the user has a key. |
| Tracing | LangSmith free tier | Local **`structlog` JSON logs** + LangGraph's built-in graph visualizer. LangSmith optional via env. |
| Vector DB / Cache / Data | Qdrant / Redis / SEC EDGAR | Unchanged — all free and self-hosted / public. |

### Tradeoffs we're accepting

1. **Absolute latency is worse.** Local 8B on CPU ≈ 10–30 tok/s; on a modest GPU it's fine. The **−35% latency claim is relative** (cached vs uncached on the same local model), so it survives — we just add a caveat in `EVALUATION.md` clarifying "measured on Ollama llama3.1:8b on <hardware spec>."
2. **Groundedness measurement is riskier.** A small local model judging its own output can bias the score. **Mitigation:** report **two** groundedness numbers in `EVALUATION.md` — (a) LLM-as-judge via TruLens, (b) NLI-entailment via a cross-encoder. If they agree at ≥94%, ship "94%". If not, ship the *lower* number verbatim (org rule: no unreproducible claims).
3. **Hardware honesty.** README states the minimum: ~8 GB free RAM for CPU inference, or any GPU with ≥6 GB VRAM. Recruiters below that bar use `DEMO_MODE=true` canned playback.
4. **Web search quality drops** (DuckDuckGo snippets vs Tavily). Web-Search Agent must sanitize + re-rank harder. Acceptable for a portfolio piece.

### Config additions (`.env.example`)

```
LLM_PROVIDER=ollama                # ollama | openai
LLM_MODEL=llama3.1:8b-instruct
OLLAMA_BASE_URL=http://ollama:11434

EMBEDDING_PROVIDER=fastembed       # fastembed | openai
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

JUDGE_MODEL=qwen2.5:14b-instruct   # for TruLens; can equal LLM_MODEL on low-RAM
NLI_MODEL=cross-encoder/nli-deberta-v3-base

WEB_SEARCH_PROVIDER=duckduckgo     # duckduckgo | tavily
TAVILY_API_KEY=                    # optional
LANGSMITH_API_KEY=                 # optional

DEMO_MODE=false
```

### Dependency deltas

- **Add:** `langchain-ollama`, `duckduckgo-search`, `sentence-transformers` (for NLI).
- **Drop from required:** `langchain-openai` moves to an optional extra (`pip install .[openai]`).
- **`docker-compose.yml`:** `ollama` service is now **on by default** (not behind a profile), with a healthcheck that waits for the model to be pulled. Add a first-run init container that runs `ollama pull llama3.1:8b-instruct` and `ollama pull qwen2.5:14b-instruct` (or a smaller judge if `LOW_RAM=true`).

### Resume claims — final defensibility check

| Claim | Status | Notes |
|---|---|---|
| Multi-agent LangGraph (Retrieval / Web-Search / Critic) | ✅ Safe | Unchanged. |
| +22% Recall@10 (hybrid vs dense) | ✅ Safe | Retrieval is model-agnostic; benchmark uses `bge-small` + BM25. |
| −35% latency (cache + streaming) | ✅ Safe with caveat | Relative metric; caveat in `EVALUATION.md` names the hardware + model. |
| ≥94% groundedness (TruLens) | ⚠️ At risk | Ship the number we actually measure. Dual-metric (LLM judge + NLI) makes whatever we report bulletproof. |

Phase 0 checklist stands, with the dep and Compose changes above folded in.

---

## 17. Hardware-locked model defaults (2026-07-30)

**Dev machine:** 32 GB RAM laptop, CPU inference (no GPU assumed).

| Role | Model | RAM (Q4_K_M) | Notes |
|---|---|---|---|
| Generator (all agents + synthesizer) | `llama3.1:8b-instruct` | ~5 GB | ~8–15 tok/s on CPU; streams acceptably over WebSocket. |
| Judge (TruLens RAG Triad) | `qwen2.5:14b-instruct` | ~10 GB | Different model family from generator → avoids self-scoring bias. Runs in batch during `make eval`. |
| NLI backup judge | `cross-encoder/nli-deberta-v3-base` | ~0.4 GB | Deterministic, CPU-fast second opinion on groundedness. |
| Dense embeddings | `BAAI/bge-small-en-v1.5` (fastembed) | ~0.1 GB | Local, no key. |
| Sparse retrieval | `Qdrant/bm25` (fastembed) | negligible | Local. |

**Peak RAM budget:** ~16 GB models + ~2 GB Qdrant/Redis/Python ≈ **~18 GB peak** (headroom: ~14 GB).

**Honest performance expectations on this hardware:**

- Interactive chat (streaming): **~25–40 s cold end-to-end, ~15–20 s warm** for a ~300-token answer.
- `make eval` (30-question TruLens run × 3 metrics with 14B judge): **~20–40 min** first run; subsequent runs cached by `(question_id, context_hash, answer_hash)` so unchanged rows are skipped.
- First-run `docker compose up` pulls **~15 GB** of Ollama models — README calls this out explicitly and offers `DEMO_MODE=true` as the zero-download path.

**`.env.example` defaults now finalized:**

```
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b-instruct
OLLAMA_BASE_URL=http://ollama:11434

EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

JUDGE_MODEL=qwen2.5:14b-instruct
NLI_MODEL=cross-encoder/nli-deberta-v3-base

WEB_SEARCH_PROVIDER=duckduckgo
TAVILY_API_KEY=
LANGSMITH_API_KEY=

DEMO_MODE=false
LOW_RAM=false                      # if true, judge falls back to qwen2.5:7b
```

Ready to scaffold Phase 0 on your say-so.
