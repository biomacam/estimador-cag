"""FastAPI dependency factories for shared singletons (cache and LLM wrapper)."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import structlog
from openai import OpenAI

from app.config import get_settings
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMConfigurationError, LLMWrapper
from app.sessions import SessionStore

if TYPE_CHECKING:
    from app.cache.semantic import EstimationSemanticCache
    from app.services.llm_service import EstimationService

log = structlog.get_logger()


@lru_cache
def get_openai_client() -> OpenAI | None:
    """Lazy OpenAI client used by input guardrails (Moderation API). ``None`` when
    no OpenAI key is configured, so moderation is skipped rather than crashing."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def get_estimation_service() -> EstimationService:
    """Build the service without resolving LLM credentials until estimation is requested."""
    from app.services.llm_service import EstimationService

    return EstimationService(
        openai_client=get_openai_client(), semantic_cache=get_semantic_cache(),
    )


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    """Build the semantic cache, swallowing setup errors so the rest of the
    pipeline keeps working if Redis Stack / RediSearch is not available
    (e.g. running on vanilla redis:7-alpine)."""
    settings = get_settings()
    openai_client = get_openai_client()
    if openai_client is None:
        log.warning("semantic_cache_disabled", reason="no_openai_key")
        return None

    try:
        import redis
        from redisvl.utils.vectorize import OpenAITextVectorizer

        from app.cache.semantic import EstimationSemanticCache

        vectorizer = OpenAITextVectorizer(
            model=settings.EMBEDDING_MODEL, api_config={"api_key": settings.OPENAI_API_KEY},
        )
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        return EstimationSemanticCache(
            redis_client=redis_client,
            vectorizer=vectorizer,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            ttl=settings.SEMANTIC_CACHE_TTL,
            log_only=settings.SEMANTIC_CACHE_LOG_ONLY,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "semantic_cache_disabled", reason="setup_failed",
            error_type=type(exc).__name__, error=str(exc)[:200],
        )
        return None


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore(max_turns=get_settings().MAX_CONVERSATION_TURNS)


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    if not settings.is_llm_configured:
        raise LLMConfigurationError(
            "No LLM provider is configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )
    return LLMWrapper(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
    )
