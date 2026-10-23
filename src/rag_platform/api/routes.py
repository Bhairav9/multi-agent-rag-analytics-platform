from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel

from rag_platform import __version__
from rag_platform.api.auth import verify_api_key

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz() -> HealthResponse:
    """Liveness probe. Returns 200 as long as the process is up."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/healthz/ready", tags=["health"], dependencies=[Depends(verify_api_key)])
async def healthz_ready(request: Request, response: Response) -> dict:
    """Readiness probe — checks Redis and Qdrant connectivity."""
    redis_status = "ok"
    qdrant_status = "ok"

    try:
        if not hasattr(request.app.state, "redis"):
            raise RuntimeError("redis not initialised")
        await request.app.state.redis.ping()
    except Exception:
        redis_status = "unreachable"

    try:
        if not hasattr(request.app.state, "qdrant_client"):
            raise RuntimeError("qdrant_client not initialised")
        request.app.state.qdrant_client.get_collections()
    except Exception:
        qdrant_status = "unreachable"

    if redis_status != "ok" or qdrant_status != "ok":
        response.status_code = 503
        return {"status": "degraded", "redis": redis_status, "qdrant": qdrant_status}

    return {"status": "ready", "redis": redis_status, "qdrant": qdrant_status}
