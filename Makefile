.PHONY: help install run test lint typecheck format ingest eval bench-latency bench-retrieval ui pull-models up down logs

help:
	@echo "Targets:"
	@echo "  install       uv sync (with dev + openai extras)"
	@echo "  run           uvicorn locally (no Docker)"
	@echo "  test          pytest with coverage"
	@echo "  lint          ruff check"
	@echo "  format        ruff format"
	@echo "  typecheck     mypy"
	@echo "  up            podman compose up -d"
	@echo "  down          podman compose down"
	@echo "  logs          podman compose logs -f app"
	@echo "  pull-models   ollama pull llama3.1:8b-instruct + qwen2.5:14b-instruct"
	@echo "  ingest          ingest SEC filings into Qdrant (TICKERS=... YEARS=...)"
	@echo "  eval            run TruLens groundedness evaluation"
	@echo "  bench-latency   cached vs uncached latency benchmark"
	@echo "  bench-retrieval dense vs hybrid recall@10 benchmark"
	@echo "  ui              run Streamlit UI locally (port 8501)"

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
	podman compose up -d --build

down:
	podman compose down

logs:
	podman compose logs -f app

pull-models:
	podman exec rag-ollama ollama pull llama3.1:8b
	podman exec rag-ollama ollama pull qwen2.5:14b

ingest:
	podman exec rag-app python -m rag_platform.ingestion.pipeline --tickers $(TICKERS) --years $(YEARS)

ingest-local:
	QDRANT_URL=http://localhost:6333 uv run python -m rag_platform.ingestion.pipeline --tickers $(TICKERS) --years $(YEARS)

eval:
	uv run python -m rag_platform.evaluation.trulens_setup

bench-latency:
	uv run python -m rag_platform.evaluation.latency_bench

bench-retrieval:
	uv run python -m rag_platform.evaluation.retrieval_bench

ui:
	uv run streamlit run src/rag_platform/ui/streamlit_app.py --server.port 8501