"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from rag_platform import __version__
from rag_platform.api.routes import router as api_router
from rag_platform.config import get_settings

def create_app() -> FastAPI:
    """Application factory"""
    settings = get_settings()
    app = FastAPI(
        title="Enterprise Multi-Agent RAG Analytics Platform",
        version=__version__,
        docs_url="/docs",
    )
    app.include_router(api_router)
    app.state.settings = settings
    return app 


app = create_app()
