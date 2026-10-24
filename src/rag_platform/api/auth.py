"""FastAPI dependency for API key authentication."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from rag_platform.config import get_settings


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if x_api_key != get_settings().api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
