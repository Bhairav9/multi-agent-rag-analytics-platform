"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    llm_provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    llm_model: str = "llama3.1:8b-instruct"
    ollama_base_url: str = "http://ollama:11434"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    embedding_provider: Literal["fastembed"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    judge_model: str = "qwen2.5:14b-instruct"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "sec_filings"

    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 86_400

    web_search_provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    tavily_api_key: str = ""

    langsmith_api_key: str = ""
    log_level: str = "INFO"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    demo_mode: bool = False
    low_ram: bool = Field(default=False, description="Falls back to smaller judge model")

    api_key: str = Field(default="change-me", description="API key for all endpoints")

    sec_edgar_user_agent: str = ""
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    retrieval_top_k: int = 10


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
