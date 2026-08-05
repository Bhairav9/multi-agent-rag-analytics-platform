# Phase 0 — Skeleton & CI Green

> Goal: end this phase with `docker compose up` starting the app, `/healthz` returning `200 OK`, `pytest` passing, and GitHub Actions green. **No RAG logic yet.** Just the plumbing.

---

## Task checklist

| # | Task | File(s) | Status |
|---|---|---|---|
| 1 | Ignore untracked / secret / data files | `.gitignore` | ⬜ |
| 2 | Declare Python project + pinned deps (uv) | `pyproject.toml` | ⬜ |
| 3 | Document every env var | `.env.example` | ⬜ |
| 4 | One-command spin-up (app + qdrant + redis + ollama) | `docker-compose.yml` | ⬜ |
| 5 | App container image | `Dockerfile` | ⬜ |
| 6 | Developer shortcuts | `Makefile` | ⬜ |
| 7 | CI (ruff + mypy + pytest + coverage) | `.github/workflows/ci.yml` | ⬜ |
| 8 | Package marker + version | `src/rag_platform/__init__.py` | ⬜ |
| 9 | Typed config loader | `src/rag_platform/config.py` | ⬜ |
| 10 | FastAPI app entrypoint | `src/rag_platform/main.py` | ⬜ |
| 11 | REST router (`/healthz`) | `src/rag_platform/api/routes.py` | ⬜ |
| 12 | LLM provider abstraction skeleton (§15 delta) | `src/rag_platform/llm/provider.py` + `__init__.py` | ⬜ |
| 13 | DEMO_MODE fixture file stub (§15 delta) | `src/rag_platform/demo/fixtures.jsonl` + `__init__.py` | ⬜ |
| 14 | Pytest fixtures | `tests/conftest.py` | ⬜ |
| 15 | First hermetic test | `tests/unit/test_health.py` | ⬜ |

Files you've already scaffolded as **empty** (agents/, retrieval/, cache/, ingestion/, evaluation/, ui/) — leave them empty in Phase 0. They get code in Phases 1–6.

---

## Core concepts (read once before you type anything)

These are the mental models you need. Each one is used somewhere in Phase 0 — the pointer tells you where.

### 1. src-layout vs flat layout

There are two ways to lay out a Python project:

**Flat layout:**
```
myproject/
├── rag_platform/     ← package at repo root
└── tests/
```

**src-layout (what we use):**
```
myproject/
├── src/
│   └── rag_platform/ ← package one level deep
└── tests/
```

**Why src-layout?** In the flat layout, `import rag_platform` accidentally works even when the package isn't installed — Python just finds the folder. This hides bugs like "forgot to add a dep." In src-layout, `import rag_platform` **only works after `pip install -e .`** — you're always testing what you'd actually ship. Modern Python (PyPA, hatchling, uv) all recommend it.

*Used in:* the whole project. The line that wires it up is `packages = ["src/rag_platform"]` in `pyproject.toml`.

---

### 2. `uv` vs `pip` / `poetry`

`uv` is a Rust-written package manager (from Astral, same people as ruff). It's a drop-in replacement for `pip` + `venv` + `poetry` + `pip-tools`.

- `uv sync` — reads `pyproject.toml`, creates `.venv/`, installs everything. 10-100× faster than pip.
- `uv run <command>` — runs `<command>` inside `.venv/` without you having to activate it.
- `uv add <package>` — adds a dependency and updates the lockfile.

**Why we pick it:** Speed matters when CI reinstalls on every commit. Also, the lockfile (`uv.lock`) guarantees the exact same versions on your machine, in CI, and in Docker.

*Used in:* every command in the Makefile and CI. Task 2, 15, 16.

---

### 3. `pyproject.toml` + build backend (hatchling)

`pyproject.toml` is the **modern replacement** for `setup.py`, `setup.cfg`, `requirements.txt`, and half of the other files you've seen in old Python projects. It's defined by PEP 517/518/621.

Three sections matter:
- `[project]` — name, version, deps, Python version.
- `[build-system]` — which library builds your wheel. We use **hatchling** (lightweight, modern).
- `[tool.*]` — every tool's config (ruff, mypy, pytest, coverage) lives here, not in separate files.

**Why one file?** No more juggling `setup.py`, `.flake8`, `pytest.ini`, `mypy.ini`, `.coveragerc`. One config, one truth.

*Used in:* Task 2.

---

### 4. `pydantic-settings` — env vars → typed objects

The problem: your app needs to read `LLM_MODEL`, `REDIS_URL`, `DEMO_MODE=true`, etc. Doing `os.environ["LLM_MODEL"]` everywhere is error-prone:
- Typos silently return `None`.
- Bool parsing is inconsistent (`"true"` vs `"True"` vs `"1"`).
- No autocomplete in your IDE.
- No validation.

`pydantic-settings` fixes this. You write:
```python
class Settings(BaseSettings):
    llm_model: str = "llama3.1:8b-instruct"
    demo_mode: bool = False
```
And now `Settings().llm_model` is typed, autocompleted, validated. `DEMO_MODE=true` in the shell auto-parses to `True`. Bad values throw `ValidationError` **at startup**, not at 3 AM in prod.

**Concept:** "boundary validation" — validate untrusted input (env, HTTP, files) once at the edge; trust it everywhere inside.

*Used in:* Task 5.

---

### 5. `@lru_cache` singleton pattern

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

The first call to `get_settings()` runs `Settings()` (reads `.env`, validates, builds object). Every subsequent call returns the **same cached object**. Effectively a lazy singleton without a global variable.

**Why this pattern instead of a module-level `settings = Settings()`?**
- Tests can call `get_settings.cache_clear()` to force re-read with different env vars.
- Import-order bugs go away — nothing runs until someone actually calls the function.
- One-line pattern with zero boilerplate.

*Used in:* Task 5.

---

### 6. FastAPI: application factory + `APIRouter`

**Naive way (bad):**
```python
app = FastAPI()

@app.get("/healthz")
def healthz(): ...
```
Everything is on one global `app`. Tests can't get a clean instance. Endpoints and app config are entangled.

**Our way — factory + router:**
```python
# routes.py
router = APIRouter()
@router.get("/healthz")
def healthz(): ...

# main.py
def create_app() -> FastAPI:
    app = FastAPI(...)
    app.include_router(router)
    return app

app = create_app()  # what uvicorn imports
```

Benefits:
- Tests do `create_app()` → fresh instance every time.
- Each domain (health, chat, sessions, metrics) gets its own router file.
- Same pattern as Flask blueprints / Express routers / Rails engines.

*Used in:* Tasks 7 and 8.

---

### 7. `TestClient` — hermetic HTTP tests

FastAPI's `TestClient` (built on `httpx`) runs your app **in-process**. When you call `client.get("/healthz")`, no real socket opens, no port is bound, no network is touched. The request goes through the ASGI stack in memory.

**Why this matters for the org rules:**
- "Unit tests must be pure and fast" → in-process = microseconds per request.
- "No network in unit tests" → literally no network.
- "Hermetic" → no side effects between test runs.

*Used in:* Tasks 11, 12.

---

### 8. `typing.Protocol` — structural typing (Go-style interfaces)

Traditional Python OOP:
```python
class OpenAIProvider(LLMProvider):  # must inherit
    ...
```

Protocol:
```python
class LLMProvider(Protocol):
    model: str
    async def astream(self, prompt: str) -> AsyncIterator[str]: ...

class OpenAIProvider:  # NO inheritance
    model = "gpt-4o-mini"
    async def astream(self, prompt): ...

def use(p: LLMProvider) -> None:  # accepts OpenAIProvider automatically
    ...
```

Any class with the right attributes/methods **automatically** satisfies the Protocol. No `class X(Interface):` needed. This is **structural typing**, the same idea as Go interfaces or TypeScript object types.

**Why we use it:** Ollama, OpenAI, Groq, vLLM clients all have different base classes. A Protocol lets us treat them uniformly without wrapping every one in an adapter.

*Used in:* Task 9.

---

### 9. Docker layer caching (the COPY order trick)

Each line in a Dockerfile creates a **layer**. Docker caches layers by hash. If the layer's inputs are unchanged, Docker skips re-running it.

**Bad Dockerfile:**
```dockerfile
COPY . /app          ← any file change busts the cache
RUN uv pip install -e .   ← reinstalls all 40 deps
```
Change one line in `main.py` → 3-minute rebuild.

**Our Dockerfile:**
```dockerfile
COPY pyproject.toml README.md ./  ← only deps manifest
RUN uv pip install --system -e .  ← cached until deps change
COPY src ./src                     ← code (cheap layer)
```
Change `main.py` → 1-second rebuild. Change `pyproject.toml` → full reinstall. Iterate fast.

*Used in:* Task 13.

---

### 10. docker-compose networking — service-name DNS

When you write `services: {app, qdrant, redis, ollama}` in `docker-compose.yml`, compose puts all four containers on a shared virtual network. Inside that network, **service names resolve to container IPs**.

That's why `.env.example` has:
```
QDRANT_URL=http://qdrant:6333       ← not localhost
REDIS_URL=redis://redis:6379/0
OLLAMA_BASE_URL=http://ollama:11434
```

The app container asks its DNS "who is `qdrant`?" and gets the Qdrant container's private IP. From *outside* Docker (your PowerShell) you'd use `localhost:6333` because port 6333 is published to the host. Two different addresses for the same service, depending on where you're calling from.

*Used in:* Tasks 3, 14.

---

### 11. 12-factor config

An industry principle from Heroku (2011) that's now table stakes. The relevant rule: **config lives in the environment, not the code**.

- ✅ `LLM_MODEL=llama3.1:8b-instruct` in `.env`
- ❌ `LLM_MODEL = "llama3.1:8b-instruct"` hardcoded in `config.py`

Why: same code artifact runs in dev, staging, prod with different env vars. No conditional `if env == "prod"` code paths. No secrets in the repo. Recruiters recognize this immediately.

*Used in:* Tasks 3, 5.

---

### 12. CI as automated enforcement

Every rule in the org instructions (≥80% coverage, no lint errors, no type errors, hermetic tests) is easy to violate accidentally. CI is the **automatic gate** that blocks the PR if any rule breaks.

Our CI runs on every push:
- `ruff check` — style + common bugs
- `ruff format --check` — formatting
- `mypy src` — type errors
- `pytest` with `--cov` and `fail_under=80` — tests + coverage

Each step is **independent** — you see *all* failures per commit, not one at a time.

*Used in:* Task 16.

---

### 13. Hermetic tests (what the org rules mean)

"Hermetic" = the test's outcome depends only on the code, not on:
- The network being reachable
- A specific system time
- Random numbers
- Files outside the test
- Which order tests run

Achieved via: `TestClient` (no real network), `respx` (mock outbound HTTP), `freezegun` (mock time), fixed random seeds, `tmp_path` fixtures.

Phase 0 has one hermetic test (`test_health.py`). Phases 1+ will have dozens.

*Used in:* Task 12 (and everything after).

---

### 14. DEMO_MODE canned playback (why it exists)

The generator model `llama3.1:8b-instruct` is 5 GB. The judge `qwen2.5:14b-instruct` is 10 GB. A recruiter with a 8 GB laptop **can't run the full stack**.

Solution: `DEMO_MODE=true` reads from `src/rag_platform/demo/fixtures.jsonl` — pre-recorded `{question, answer, citations}` triples — and streams them over the WebSocket as if the LLM produced them. Same UX, no compute.

**Phase 0 just creates the empty stub file.** Fixtures land in Phase 6.5. But the file has to exist now so Docker builds don't fail.

*Used in:* Task 10.

---

Now you have the concepts. The task descriptions below tell you **what** to do; the code walkthroughs after tell you **how**. Refer back here whenever a task's "Concept to understand" line feels like a black box.

---

## Order to type things

Follow this order — each step verifies the previous one works.

1. `.gitignore` → so you don't accidentally commit secrets/data now.
2. `pyproject.toml` → so `uv sync` works.
3. `.env.example` → so `config.py` has something to point at.
4. `src/rag_platform/__init__.py`, `config.py`, `api/routes.py`, `main.py` → the app itself.
5. `src/rag_platform/llm/__init__.py` + `provider.py` (empty stub).
6. `src/rag_platform/demo/__init__.py` + `fixtures.jsonl` (empty stub).
7. `tests/conftest.py`, `tests/unit/test_health.py` → prove the app works locally.
8. `Dockerfile` → containerize.
9. `docker-compose.yml` → orchestrate services.
10. `Makefile` → shortcuts.
11. `.github/workflows/ci.yml` → CI green.

---

## Task descriptions (what you're actually doing, and why)

Before you touch any code, read this section. It explains **what each task accomplishes**, **why the project needs it**, and **how you'll know it's done**. The code blocks come after — treat them as the reference implementation once you understand the intent.

---

### Task 1 — Create `.gitignore`

**What you're doing:** Telling Git which files never to track: your local secrets (`.env`), the downloaded SEC PDFs (large, regenerable), Python bytecode, virtualenvs, model caches, and IDE junk.

**Why the project needs it:** The org rules say *no secrets in code*. If you commit a `.env` once with a real API key (even a free-tier Tavily key), it's in Git history forever. Also, `data/raw/` will hold ~50–150 MB of filings — you don't want that in the repo.

**Trick to know:** The `!data/raw/.gitkeep` pattern **re-includes** a specific file inside an otherwise-ignored folder. That's how empty directory shells survive in Git (Git tracks files, not folders).

**Done when:** `git status` doesn't list `.env`, `__pycache__/`, or files under `data/raw/` even after you create them.

---

### Task 2 — Create `pyproject.toml`

**What you're doing:** Declaring the project name, version, Python version, all runtime dependencies (LangGraph, Qdrant, Redis, FastAPI, etc.), all dev dependencies (pytest, ruff, mypy), and every tool config (linter rules, type-checker strictness, coverage threshold).

**Why the project needs it:** This is the *single source of truth* for reproducibility. Any recruiter running `uv sync` gets the exact same environment you built on. It also enforces the org rule "≥80% coverage" via `[tool.coverage.report].fail_under = 80` — CI will literally fail the build if coverage drops.

**Concept to understand:** `[build-system] = hatchling` + `[tool.hatch.build.targets.wheel].packages = ["src/rag_platform"]` is what makes the **src-layout** work. Without those two lines, `import rag_platform` wouldn't find your code.

**Optional extras explained:**
- `pip install .[dev]` → pytest, ruff, mypy (only devs need these)
- `pip install .[openai]` → adds `langchain-openai` (only if you have an OpenAI key). Per §16, this is not the default.

**Done when:** `uv sync --all-extras` completes without errors and creates `.venv/` with all packages installed.

---

### Task 3 — Create `.env.example`

**What you're doing:** Documenting every environment variable the app reads, with a safe default value (or blank for secrets). This file **is committed**; the real `.env` (copied from it) never is.

**Why the project needs it:** 12-factor rule — config in env, not code. Also, this file is how a new developer (or future you) discovers what knobs exist without reading source code. Section 10 of discuss.md lists this as a non-functional requirement.

**Trick to know:** Every URL uses **Docker service names** as hostnames (`http://qdrant:6333`, `http://ollama:11434`, `redis://redis:6379`). Inside the compose network, service names resolve to their container IPs. If you're running locally *without* Docker, you'd swap these to `localhost`.

**Done when:** Every setting in `config.py` has a matching line here, and no real secret is written in.

---

### Task 4 — Create `src/rag_platform/__init__.py`

**What you're doing:** Turning the `rag_platform` folder into an actual importable Python package and exposing a `__version__` string.

**Why the project needs it:** Without this file, Python doesn't treat the folder as a package (in strict src-layout with hatchling). Also, `__version__` is what `/healthz` returns — it's how you'll prove which build is deployed.

**Done when:** `python -c "import rag_platform; print(rag_platform.__version__)"` prints `0.1.0`.

---

### Task 5 — Create `src/rag_platform/config.py`

**What you're doing:** Building a **typed settings object** using `pydantic-settings` that reads `.env` and exposes every config value as a strongly-typed attribute (e.g., `settings.llm_provider: Literal["ollama", "openai"]`).

**Why the project needs it:** This is the *boundary* between the outside world (env vars, files) and your typed Python code. If someone sets `LLM_PROVIDER=grok` (typo), the app **fails at startup** with a clear `ValidationError` — not silently at 3 AM in production.

**Concept to understand:** `@lru_cache(maxsize=1)` on `get_settings()`. This means the `Settings()` object is constructed exactly **once per process**. Every module across your app calls `get_settings()` and gets the same instance — no re-reading `.env`, no duplicated validation.

**Done when:** In a Python REPL, `from rag_platform.config import get_settings; get_settings().llm_model` returns `'llama3.1:8b-instruct'`.

---

### Task 6 — Create `src/rag_platform/api/__init__.py`

**What you're doing:** Marking the `api/` subfolder as a subpackage.

**Why:** Same as Task 4, but for the `api` subpackage. One-line docstring is enough.

**Done when:** `from rag_platform.api.routes import router` works.

---

### Task 7 — Create `src/rag_platform/api/routes.py`

**What you're doing:** Defining your first HTTP endpoint — `GET /healthz` — using FastAPI's `APIRouter`. It returns a JSON body `{"status": "ok", "version": "0.1.0"}` with HTTP 200.

**Why the project needs it:** Three reasons:
1. **Docker healthcheck** — `docker-compose.yml` uses `/healthz` to know when the container is ready before dependent services start.
2. **CI smoke test** — `tests/unit/test_health.py` hits this to prove the app boots.
3. **Modular routing pattern** — every future endpoint (chat, sessions, metrics) will register on its own router and get mounted in `main.py`. Keeps concerns separated.

**Concept to understand:** `APIRouter` vs putting routes on `FastAPI` directly. Routers are just route containers — they let you split endpoints across files and test them in isolation. Same pattern as Flask blueprints or Express routers.

**Done when:** After Tasks 8+15, `curl http://localhost:8000/healthz` returns the JSON body.

---

### Task 8 — Create `src/rag_platform/main.py`

**What you're doing:** Building the FastAPI **application factory** — a function `create_app()` that constructs a fresh `FastAPI` instance, loads settings, and mounts every router. Then exposing a module-level `app = create_app()` for uvicorn to import.

**Why the project needs it:** Two roles in one file:
1. **Production entrypoint** — uvicorn imports `rag_platform.main:app`.
2. **Test-friendly factory** — tests call `create_app()` to build isolated instances (different settings, mock dependencies later).

**Concept to understand:** **Application factory pattern**. Never keep app state in module-level globals other than `app` itself; put it on `app.state.settings`. This is what makes the app testable — every `pytest` gets a clean instance.

**Done when:** `uv run uvicorn rag_platform.main:app --reload` starts and prints `Uvicorn running on http://0.0.0.0:8000`.

---

### Task 9 — Create `src/rag_platform/llm/__init__.py` + `provider.py`

**What you're doing:** Creating an **empty stub** for the LLM abstraction layer (per §15 delta). Just enough type structure so future imports don't break.

**Why the project needs it:** Section 16 of discuss.md pivoted to zero-paid-services (Ollama default). But the resume claim is "provider-agnostic design." A `Protocol` interface here declares that any future backend (OpenAI, Groq, vLLM) must implement `.astream(prompt) -> AsyncIterator[str]`. The real implementation lands in Phase 2 — this is scaffolding.

**Concept to understand:** `typing.Protocol` = structural typing. Any class with a `model: str` attribute and an `astream()` coroutine automatically satisfies `LLMProvider` — no explicit inheritance needed. It's Python's version of Go interfaces.

**Done when:** `from rag_platform.llm.provider import LLMProvider` works. No functional test yet.

---

### Task 10 — Create `src/rag_platform/demo/__init__.py` + `fixtures.jsonl`

**What you're doing:** Creating another **empty stub** for the DEMO_MODE playback path (per §15 delta). `fixtures.jsonl` is a literally empty file; you'll fill it with `{question, canned_answer, canned_citations}` triples in Phase 6.5.

**Why the project needs it:** Recruiters without 15 GB free to pull Ollama models need a working chat. `DEMO_MODE=true` reads from this file and replays pre-recorded answers over the WebSocket — no LLM call at all. This is what makes the "one-command demo" claim in §10 true even for cold hardware.

**Done when:** The file exists. Nothing to test yet.

---

### Task 11 — Create `tests/conftest.py`

**What you're doing:** Defining the **shared `client` fixture** — a `TestClient` wrapping a fresh app instance. Every test that needs to hit an endpoint just declares `def test_x(client):` and gets a working in-process HTTP client.

**Why the project needs it:** Org rule: "Unit tests must be pure and fast; use mocks/fakes for IO." `TestClient` runs FastAPI **in-process** — no real port, no real network. Tests execute in milliseconds and are 100% deterministic. Also, DRY — every route test would otherwise re-instantiate the app.

**Concept to understand:** `pytest.fixture` + `yield` = setup/teardown pattern. Code before `yield` runs before the test; code after `yield` runs after. The `with TestClient(app) as ...` block gives you proper connection cleanup automatically.

**Done when:** After Task 12, running `uv run pytest` finds and uses the fixture without errors.

---

### Task 12 — Create `tests/unit/test_health.py`

**What you're doing:** Writing two assertions: (1) `GET /healthz` returns 200 with the exact JSON shape, (2) `GET /openapi.json` returns 200 with the right app title. This is your first **hermetic** test — no network, no external services, runs in <100 ms.

**Why the project needs it:** The org rules mandate TDD + ≥80% coverage. This test also **gets CI green** — the whole point of Phase 0. And it's a canary: if some future refactor breaks app boot or drops `/healthz`, this test catches it in the PR.

**Trick to know:** Test 2 (openapi schema) is a proxy check for "did the app initialize cleanly?" If any router failed to mount or any dependency failed to resolve, the schema call would 500. Cheap, powerful signal.

**Done when:** `uv run pytest -q` prints `2 passed`.

---

### Task 13 — Create `Dockerfile`

**What you're doing:** Writing a container recipe: start from `python:3.11-slim`, install `uv`, copy dependency manifest first, install deps, then copy source, expose port 8000, and set `uvicorn` as the default command.

**Why the project needs it:** Section 10 requires `docker compose up` = one-command bring-up. Also, containerization is table stakes for the "Production hygiene" bullet in §1.

**Concept to understand:** **Layer caching**. Docker caches each `RUN`/`COPY` step. By copying `pyproject.toml` *first* and installing deps *before* copying `src/`, you get: dep install only reruns when deps change. Change a Python file → 1-second rebuild. Change `pyproject.toml` → full reinstall. Big win when iterating.

**Done when:** `docker build -t rag-app .` completes without errors.

---

### Task 14 — Create `docker-compose.yml`

**What you're doing:** Declaring four services — `app` (your FastAPI), `qdrant`, `redis`, `ollama` — with named volumes for persistence, port mappings, and dependency ordering. Per §16, Ollama is now on by default (not behind a profile).

**Why the project needs it:** The killer demo from §2 ("summarize JPMorgan's exposure and cross-check with news") needs *all* four services alive simultaneously. Compose is what makes that a single command instead of four terminal windows.

**Trick to know:** Model pulling is *deferred* to `make pull-models`. If you tried to pull `llama3.1:8b-instruct` + `qwen2.5:14b-instruct` on first `docker compose up`, you'd wait 15+ GB of download before anything responds. Splitting it out means the shell exits fast; you pull models when *you* decide.

**Done when:** `docker compose up -d` starts all four containers and `docker compose ps` shows them healthy.

---

### Task 15 — Create `Makefile`

**What you're doing:** Wrapping every common command (install, run, test, lint, up/down, pull-models, ingest, eval) as a one-word `make <target>` shortcut.

**Why the project needs it:** Section 10 NFRs explicitly call for `make ingest`, `make eval`, `make test`. Recruiters cloning the repo shouldn't have to memorize `uv run pytest --cov=... --cov-report=...`. They just type `make test`.

**Windows note:** PowerShell has no `make`. Either install with `winget install GnuWin32.Make` OR run the underlying `uv run ...` commands directly — the Makefile then serves as documentation for what to type.

**Done when:** `make help` prints the target list.

---

### Task 16 — Create `.github/workflows/ci.yml`

**What you're doing:** Defining a GitHub Actions job that runs on every push and pull request to `main`. Steps: check out code → set up Python 3.11 → install uv → install deps → run ruff → run mypy → run pytest with coverage → upload coverage XML as an artifact.

**Why the project needs it:** Every claim on the resume must be reproducible. CI is the mechanism that *automatically enforces* the org rules on every commit: no lint errors, no type errors, no failing tests, no coverage drop below 80%. Also, a green checkmark on your GitHub PRs is what a recruiter sees at a glance.

**Concept to understand:** Each step is **independent** — a ruff failure doesn't skip mypy, so you see *all* problems in one CI run instead of playing whack-a-mole. This is why the workflow has separate `Lint`, `Type check`, `Test` steps instead of one big `make ci`.

**Done when:** You push to GitHub and the Actions tab shows a green ✓.

---

## File-by-file walkthrough

### 1. `.gitignore`

**Why:** Prevent secrets (`.env`), raw datasets, model caches, virtualenvs, and IDE junk from ever hitting Git. `.gitkeep` files are re-included so empty directory shells stay in the repo.

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
*.egg
.venv/
venv/
env/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage
.coverage.*
coverage.xml
dist/
build/

# Env / secrets
.env
.env.local
.env.*.local

# Data (large, regenerable)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Model / embed caches
.cache/
.fastembed_cache/
.ollama/

# IDE / OS
.vscode/
.idea/
*.swp
.DS_Store
Thumbs.db

# TruLens local DB
default.sqlite
trulens/
```

**Key lines to understand:**
- `.env` blocked, `.env.example` allowed → secrets never committed, but the template is.
- `data/raw/*` blocked but `!data/raw/.gitkeep` re-included → folder shape survives, contents don't.

---

### 2. `pyproject.toml`

**Why:** Single source of truth for dependencies, tool config (ruff, mypy, pytest, coverage), and build metadata. Managed by [`uv`](https://docs.astral.sh/uv/).

```toml
[project]
name = "rag-platform"
version = "0.1.0"
description = "Enterprise Multi-Agent RAG Analytics Platform over SEC filings"
readme = "README.md"
requires-python = ">=3.11,<3.13"
license = { text = "MIT" }
authors = [{ name = "You" }]

dependencies = [
    # Web
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "websockets>=13.1",

    # LangGraph / LangChain
    "langgraph>=0.2.40",
    "langchain>=0.3.7",
    "langchain-core>=0.3.15",
    "langchain-ollama>=0.2.0",
    "langchain-community>=0.3.5",

    # Vector store + embeddings
    "qdrant-client>=1.12.0",
    "fastembed>=0.4.1",

    # Cache
    "redis>=5.2.0",

    # Config + logging
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    "structlog>=24.4.0",

    # UI
    "streamlit>=1.40.0",

    # Ingestion
    "pymupdf>=1.24.13",
    "beautifulsoup4>=4.12.3",
    "httpx>=0.27.2",

    # Web search
    "duckduckgo-search>=6.3.5",

    # Evaluation
    "trulens-eval>=0.33.0",
    "sentence-transformers>=3.3.0",
]

[project.optional-dependencies]
openai = ["langchain-openai>=0.2.5"]
dev = [
    "pytest>=8.3.3",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "respx>=0.21.1",
    "httpx>=0.27.2",
    "ruff>=0.7.4",
    "mypy>=1.13.0",
    "types-redis>=4.6.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_platform"]

# --- Ruff -----------------------------------------------------------------
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "C4"]
ignore = ["E501"]  # line-length handled by formatter

# --- Mypy -----------------------------------------------------------------
[tool.mypy]
python_version = "3.11"
strict = true
mypy_path = "src"
packages = ["rag_platform"]
ignore_missing_imports = true

# --- Pytest ---------------------------------------------------------------
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra --strict-markers --cov=rag_platform --cov-report=term-missing --cov-report=xml"

# --- Coverage -------------------------------------------------------------
[tool.coverage.run]
branch = true
source = ["src/rag_platform"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

**Key sections to understand:**
- `[project.dependencies]` — runtime deps. **No `langchain-openai`** — moved to optional `[openai]` extra per §16.
- `[project.optional-dependencies].dev` — everything you need for testing/linting only.
- `[build-system] = hatchling` — modern PEP 517 backend, works cleanly with uv.
- `[tool.hatch.build.targets.wheel].packages = ["src/rag_platform"]` — this is the line that makes the src-layout work.
- `[tool.pytest.ini_options].addopts` — every `pytest` run measures coverage and fails if a marker is unknown.
- `[tool.coverage.report].fail_under = 80` — enforces the org rule.

After you write this file, run once:
```powershell
uv sync --all-extras
```
That creates `.venv/` and installs everything.

---

### 3. `.env.example`

**Why:** Documents every env var the app reads. Copied to `.env` by the developer; never committed.

```dotenv
# --- LLM (generator) ---
LLM_PROVIDER=ollama                # ollama | openai
LLM_MODEL=llama3.1:8b-instruct
OLLAMA_BASE_URL=http://ollama:11434

# --- Embeddings ---
EMBEDDING_PROVIDER=fastembed       # fastembed | openai
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# --- Evaluation (TruLens) ---
JUDGE_MODEL=qwen2.5:14b-instruct
NLI_MODEL=cross-encoder/nli-deberta-v3-base

# --- Vector store ---
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=sec_filings

# --- Cache ---
REDIS_URL=redis://redis:6379/0
CACHE_TTL_SECONDS=86400

# --- Web search ---
WEB_SEARCH_PROVIDER=duckduckgo     # duckduckgo | tavily
TAVILY_API_KEY=

# --- Observability (optional) ---
LANGSMITH_API_KEY=
LOG_LEVEL=INFO

# --- App / demo ---
APP_HOST=0.0.0.0
APP_PORT=8000
DEMO_MODE=false
LOW_RAM=false                      # if true, judge falls back to qwen2.5:7b

# --- OpenAI (optional, only if LLM_PROVIDER=openai) ---
OPENAI_API_KEY=
```

**Key lines:**
- Every URL uses **Docker service names** (`ollama`, `qdrant`, `redis`) — inside the compose network, hostnames resolve to services.
- Every optional key is present but blank — makes it obvious what CAN be set without needing to read code.

---

### 4. `src/rag_platform/__init__.py`

**Why:** Marks the package + exposes the version string.

```python
"""Enterprise Multi-Agent RAG Analytics Platform."""

__version__ = "0.1.0"
```

---

### 5. `src/rag_platform/config.py`

**Why:** Single typed settings object loaded from `.env` at startup. Fail-fast on bad config. This is the 12-factor "config in env" boundary.

```python
"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_model: str = "llama3.1:8b-instruct"
    ollama_base_url: str = "http://ollama:11434"
    openai_api_key: str = ""

    # Embeddings
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Evaluation
    judge_model: str = "qwen2.5:14b-instruct"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # Vector store
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "sec_filings"

    # Cache
    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 86_400

    # Web search
    web_search_provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    tavily_api_key: str = ""

    # Observability
    langsmith_api_key: str = ""
    log_level: str = "INFO"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    demo_mode: bool = False
    low_ram: bool = Field(default=False, description="Falls back to smaller judge model")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
```

**Key concepts:**
- `SettingsConfigDict(env_file=".env")` — pydantic-settings reads `.env` automatically.
- `Literal["ollama", "openai"]` — invalid values raise `ValidationError` at import time. Fail-fast.
- `@lru_cache(maxsize=1)` — instantiate once per process. Every module calls `get_settings()`; the actual object is built once.
- No secrets hardcoded — everything defaults to a safe empty string.

---

### 6. `src/rag_platform/api/__init__.py`

```python
"""HTTP API layer (REST + WebSocket)."""
```

---

### 7. `src/rag_platform/api/routes.py`

**Why:** REST endpoints. Phase 0 only ships `/healthz` for the container healthcheck + CI smoke test.

```python
"""REST routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from rag_platform import __version__

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz() -> HealthResponse:
    """Liveness probe. Returns 200 as long as the process is up."""
    return HealthResponse(status="ok", version=__version__)
```

**Key concepts:**
- `APIRouter` — modular; `main.py` mounts it. Keeps routes decoupled from app construction (easier to unit test).
- `response_model=HealthResponse` — FastAPI validates + documents the response shape in OpenAPI.
- `async def` — matches the FastAPI async style you'll use everywhere later.

---

### 8. `src/rag_platform/main.py`

**Why:** FastAPI application object. This is what `uvicorn` runs.

```python
"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from rag_platform import __version__
from rag_platform.api.routes import router as api_router
from rag_platform.config import get_settings


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    app = FastAPI(
        title="Enterprise Multi-Agent RAG Analytics Platform",
        version=__version__,
        docs_url="/docs",
    )
    app.include_router(api_router)
    # settings is bound for future dependency-injection use
    app.state.settings = settings
    return app


app = create_app()
```

**Key concepts:**
- **Application factory pattern** (`create_app()`): lets tests build fresh app instances with different settings.
- `app.state.settings` — the standard FastAPI way to attach app-wide singletons.
- Module-level `app = create_app()` — what `uvicorn rag_platform.main:app` imports.

---

### 9. `src/rag_platform/llm/__init__.py`

```python
"""LLM provider abstraction (Ollama default, OpenAI optional)."""
```

### 10. `src/rag_platform/llm/provider.py`

**Why:** Stub for now — §15 delta. Real implementation lands in Phase 2. Just declare the interface so future imports resolve.

```python
"""LLM provider abstraction. Filled in Phase 2."""

from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    model: str

    async def astream(self, prompt: str) -> "AsyncIterator[str]":  # noqa: F821
        """Yield response tokens as they are generated."""
        ...
```

---

### 11. `src/rag_platform/demo/__init__.py`

```python
"""DEMO_MODE canned-playback path for key-less recruiters."""
```

### 12. `src/rag_platform/demo/fixtures.jsonl`

Empty for now. Real fixtures land in Phase 6.5.

```
```
(Literally zero bytes. Just create the file.)

---

### 13. `tests/conftest.py`

**Why:** Shared pytest fixtures. In Phase 0 we only need a FastAPI test client.

```python
"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_platform.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Fresh FastAPI test client per test."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
```

**Key concepts:**
- Uses the **factory** `create_app()`, not the module-level `app` — so each test gets a clean instance.
- `TestClient` runs FastAPI in-process; no network, no ports, hermetic.

---

### 14. `tests/unit/test_health.py`

**Why:** First real test. Gets CI green. Verifies the app boots and `/healthz` returns the contract.

```python
"""Health-check endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_platform import __version__


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "version": __version__}


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"].startswith("Enterprise Multi-Agent")
```

**Key concepts:**
- Two assertions, not one — proves the endpoint AND the OpenAPI schema (a proxy for "app booted cleanly").
- No mocks needed — nothing external is called in `/healthz`.

Run locally now:
```powershell
uv run pytest -q
```
You should see `2 passed`.

---

### 15. `Dockerfile`

**Why:** Reproducible app container. Uses `uv` for fast dep install.

```dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# uv (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Cache deps: copy only pyproject first
COPY pyproject.toml README.md ./
RUN uv pip install --system -e .

# Now copy source
COPY src ./src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "rag_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key concepts:**
- **Two-step COPY**: `pyproject.toml` first → cache dep install layer. Source copied later → code changes don't reinstall deps.
- `uv pip install --system -e .` — editable install so container reflects the layout.
- `HEALTHCHECK` — Docker knows when the app is ready; other services can `depends_on: condition: service_healthy`.

---

### 16. `docker-compose.yml`

**Why:** One command spins up everything. Per §16, Ollama is on by default.

```yaml
services:
  app:
    build: .
    container_name: rag-app
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      qdrant:
        condition: service_started
      redis:
        condition: service_started
      ollama:
        condition: service_started
    volumes:
      - ./src:/app/src
      - ./data:/app/data

  qdrant:
    image: qdrant/qdrant:v1.12.4
    container_name: rag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage

  redis:
    image: redis:7.4-alpine
    container_name: rag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  ollama:
    image: ollama/ollama:0.4.4
    container_name: rag-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    # Pull models on demand via `make pull-models` — keeps first `up` fast.

volumes:
  qdrant_storage:
  redis_data:
  ollama_data:
```

**Key concepts:**
- `env_file: .env` — the app reads secrets/config from your local `.env`, never baked into the image.
- Named volumes (`qdrant_storage`, `ollama_data`) — data survives `docker compose down`.
- No model pull at startup — deferred to a Makefile target so `docker compose up` isn't a 15 GB download.

---

### 17. `Makefile`

**Why:** Developer shortcuts. Matches the "One-command X" NFRs in §10.

```makefile
.PHONY: help install run test lint typecheck format ingest eval pull-models up down logs

help:
	@echo "Targets:"
	@echo "  install       uv sync (with dev + openai extras)"
	@echo "  run           uvicorn locally (no Docker)"
	@echo "  test          pytest with coverage"
	@echo "  lint          ruff check"
	@echo "  format        ruff format"
	@echo "  typecheck     mypy"
	@echo "  up            docker compose up -d"
	@echo "  down          docker compose down"
	@echo "  logs          docker compose logs -f app"
	@echo "  pull-models   ollama pull llama3.1:8b-instruct + qwen2.5:14b-instruct"
	@echo "  ingest        Phase 1 stub"
	@echo "  eval          Phase 5 stub"

install:
	uv sync --all-extras

run:
	uv run uvicorn rag_platform.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f app

pull-models:
	docker exec rag-ollama ollama pull llama3.1:8b-instruct
	docker exec rag-ollama ollama pull qwen2.5:14b-instruct

ingest:
	@echo "Phase 1: python -m rag_platform.ingestion.pipeline"

eval:
	@echo "Phase 5: python -m rag_platform.evaluation.retrieval_bench && ..."
```

**Windows note:** PowerShell doesn't have `make` natively. Install with `winget install GnuWin32.Make` OR call the underlying `uv run ...` commands directly.

---

### 18. `.github/workflows/ci.yml`

**Why:** Every push runs lint + type-check + tests + coverage gate. This is the "CI green" bar.

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "0.5.4"

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Lint (ruff)
        run: uv run ruff check .

      - name: Format check (ruff)
        run: uv run ruff format --check .

      - name: Type check (mypy)
        run: uv run mypy src

      - name: Test + coverage
        run: uv run pytest

      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
```

**Key concepts:**
- Runs on every push + PR to `main`.
- **Every gate is independent** — a lint failure doesn't hide a type failure; you see all errors.
- Coverage XML uploaded so you can wire in Codecov later if you want a badge.

---

## Verification (do these in order)

```powershell
# 1. Install
uv sync --all-extras

# 2. Local test
uv run pytest -q
# Expect: 2 passed

# 3. Local run
uv run uvicorn rag_platform.main:app --reload
# In another shell:
Invoke-WebRequest http://localhost:8000/healthz | Select-Object -ExpandProperty Content
# Expect: {"status":"ok","version":"0.1.0"}

# 4. Lint + types
uv run ruff check .
uv run mypy src

# 5. Docker
Copy-Item .env.example .env
docker compose up -d --build
Invoke-WebRequest http://localhost:8000/healthz
docker compose down

# 6. Commit + push → watch CI
git add .
git commit -m "phase-0: skeleton, docker, ci"
git push
```

When all six pass, tag Phase 0 done:
```powershell
git tag phase-0-done
git push --tags
```

---

## What Phase 0 explicitly does NOT include

- Any LLM / Qdrant / Redis client code — those files stay empty.
- Any LangGraph nodes.
- Any ingestion.
- Any streaming.
- Any TruLens.

Phase 0's only job is: **a running FastAPI app in Docker with green tests.** Everything else is later phases.

Next up: **Phase 1 — Ingestion + Retrieval** (SEC EDGAR downloader → chunker → Qdrant hybrid index).
