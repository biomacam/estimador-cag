from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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

    # --- History compression (cumulative summary + anchors) ---
    # "heuristic" (regex, no LLM call) or "llm" (one classifier call per evicted turn).
    ANCHOR_DETECTION_MODE: Literal["heuristic", "llm"] = "heuristic"
    # Cheap model for the cumulative summarizer and the LLM anchor classifier.
    COMPRESSION_MODEL: str = "gpt-4o-mini"

    # --- Actor-Critic-Boss ---
    # Read-only auditor of each draft; a cheap model is enough.
    CRITIC_MODEL: str = "gpt-4o-mini"
    # Each iteration = 1 actor + 1 critic call. Three = initial draft + two directed retries.
    BOSS_MAX_ITERATIONS: int = Field(default=3, ge=1, le=5)

    # --- Ingestion (Postgres persistence + data-source catalog + PII) ---
    DATABASE_URL: str = "postgresql+psycopg://estimator:estimator@localhost:5433/estimator"
    # Resolved relative to the working directory.
    CATALOG_PATH: Path = Path("data/catalog/catalog.yaml")
    # Root that ``CatalogSource.location`` entries are resolved against.
    INGESTION_DATA_ROOT: Path = Path("data/seed")
    # Must be a Spanish model: Presidio's default silently misses Hispanic names.
    PRESIDIO_SPACY_MODEL: str = "es_core_news_md"
    PSEUDONYM_FAKER_LOCALE: str = "es_ES"
    # HMAC salt behind pseudonym_mappings.original_hash; rotating it invalidates all mappings.
    PSEUDONYM_HASH_SALT: str = "change-me-in-prod"

    @model_validator(mode="after")
    def _salt_must_be_set_in_production(self) -> "Settings":
        if self.APP_ENV == "production" and self.PSEUDONYM_HASH_SALT == "change-me-in-prod":
            raise ValueError("PSEUDONYM_HASH_SALT must be changed from its default in production.")
        return self

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
