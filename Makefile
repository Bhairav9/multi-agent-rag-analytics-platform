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