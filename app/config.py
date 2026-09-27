from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    ESTIMATOR_API_BASE_URL: str = "http://localhost:8000"

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
