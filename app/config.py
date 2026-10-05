from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.sessions import MAX_TURNS


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"

    # --- Session 3 fields (LiteLLM wrapper, Redis cache, Streamlit transport) ---
    PRIMARY_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2

    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400

    # --- Semantic cache (composite key: deterministic bucket + vector similarity) ---
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_TTL: int = 86400
    # When True, the semantic cache LOGS potential hits but does NOT serve them.
    # Used to gather metrics before flipping the cache on in production.
    SEMANTIC_CACHE_LOG_ONLY: bool = False

    ESTIMATOR_API_BASE_URL: str = "http://localhost:8000"

    # --- Session attachments (local PDF/DOCX text extraction) ---
    MAX_ATTACHMENTS: int = 5
    MAX_ATTACHMENT_BYTES: int = 10 * 1024 * 1024
    # Per file, after extraction: bounds the tokens a single upload can add to the prompt.
    MAX_ATTACHMENT_CHARS: int = 60_000

    # --- Conversational sessions ---
    # user+assistant pairs kept in each session's history window.
    MAX_CONVERSATION_TURNS: int = Field(default=MAX_TURNS, ge=1)
    # Cheap model for the per-turn ProjectMetadata extraction (needs OPENAI_API_KEY).
    METADATA_EXTRACTOR_MODEL: str = "gpt-4o-mini"

    @property
    def is_llm_configured(self) -> bool:
        """True when at least one provider API key is set.

        Intentionally NOT enforced at startup: missing credentials must surface as a
        503 on the estimation endpoints, not as a crash before /health can respond.
        """
        return bool(self.OPENAI_API_KEY or self.ANTHROPIC_API_KEY)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
