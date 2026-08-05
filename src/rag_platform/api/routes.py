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
