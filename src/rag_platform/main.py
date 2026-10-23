"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import QdrantClient

from rag_platform import __version__
from rag_platform.agents.graph import build_graph
from rag_platform.api.routes import router as api_router
from rag_platform.api.websocket import ws_router
from rag_platform.cache.redis_client import get_redis_client
from rag_platform.config import get_settings
from rag_platform.llm.provider import get_llm_provider
from rag_platform.logging_config import setup_logging
from rag_platform.retrieval.bm25 import FastEmbedSparse
from rag_platform.retrieval.embeddings import FastEmbedDense
from rag_platform.retrieval.hybrid import HybridRetriever
from rag_platform.retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Application starting up", extra={"version": __version__})

    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", "rag-platform")

    # Skip heavy initialization in test mode (no Qdrant/Ollama/Redis available).
    if not os.getenv("TESTING"):
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        store = QdrantStore(qdrant_client, settings.qdrant_collection)
        retriever = HybridRetriever(FastEmbedDense(), FastEmbedSparse(), store)
        llm = get_llm_provider()
        redis = get_redis_client()
        graph = await build_graph(llm, retriever)
        app.state.graph = graph
        app.state.redis = redis
        app.state.qdrant_client = qdrant_client  # reuse — don't open two connections
        logger.info("Application startup complete")
    yield
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Application factory"""
    settings = get_settings()
    app = FastAPI(
        title="Enterprise Multi-Agent RAG Analytics Platform",
        version=__version__,
        docs_url="/docs",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    app.include_router(ws_router)
    app.state.settings = settings
    import os
    if not os.getenv("TESTING"):
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    return app


app = create_app()
