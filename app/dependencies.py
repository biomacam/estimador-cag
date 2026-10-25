"""FastAPI dependency factories for shared singletons (cache and LLM wrapper)."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import structlog
from openai import OpenAI

from app.config import get_settings
from app.embedding_pipeline.base import Chunker
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.ingestion.catalog import DataCatalog, load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import ParserRegistry, default_registry
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


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    """Shared stateless structural chunker."""
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder | None:
    """Return no embedder when an OpenAI key is not configured."""
    settings = get_settings()
    client = get_openai_client()
    if client is None:
        log.warning("embedder_disabled", reason="no_openai_key")
        return None
    return OpenAIEmbedder(client=client, model=settings.EMBEDDING_MODEL)


@lru_cache
def get_anthropic_client():
    settings = get_settings()
    if not settings.ANTHROPIC_API_KEY:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


# Strategy modules are imported lazily: langchain/nltk are slow to load at startup.
def _fixed_size() -> Chunker:
    from app.embedding_pipeline.strategies import FixedSizeChunker

    return FixedSizeChunker()


def _recursive() -> Chunker:
    from app.embedding_pipeline.strategies import RecursiveChunker

    return RecursiveChunker()


def _sentence_window() -> Chunker:
    from app.embedding_pipeline.strategies import SentenceWindowChunker

    return SentenceWindowChunker()


def _semantic() -> Chunker:
    from app.embedding_pipeline.strategies import SemanticChunker

    settings = get_settings()
    return SemanticChunker(api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL)


def _propositional() -> Chunker:
    from app.embedding_pipeline.strategies import PropositionalChunker

    client = get_openai_client()
    if client is None:
        raise RuntimeError("PropositionalChunker requires OPENAI_API_KEY.")
    return PropositionalChunker(client=client)


def _contextual_retrieval() -> Chunker:
    from app.embedding_pipeline.strategies import ContextualRetrievalChunker

    client = get_anthropic_client()
    if client is None:
        raise RuntimeError("ContextualRetrievalChunker requires ANTHROPIC_API_KEY.")
    return ContextualRetrievalChunker(client=client)


def _hierarchical() -> Chunker:
    from app.embedding_pipeline.strategies import HierarchicalChunker

    return HierarchicalChunker()


# Order is the canonical comparison order used when no strategy is requested.
CHUNKER_FACTORIES = {
    "structural": get_chunker,
    "fixed_size": _fixed_size,
    "recursive": _recursive,
    "sentence_window": _sentence_window,
    "semantic": _semantic,
    "propositional": _propositional,
    "contextual_retrieval": _contextual_retrieval,
    "hierarchical": _hierarchical,
}
ALL_STRATEGIES = list(CHUNKER_FACTORIES)


def build_chunkers(names: list[str]) -> list[Chunker]:
    """Raises ``KeyError`` for an unknown strategy, ``RuntimeError`` for a missing API key."""
    chunkers: list[Chunker] = []
    for name in names:
        factory = CHUNKER_FACTORIES.get(name)
        if factory is None:
            raise KeyError(name)
        chunkers.append(factory())
    return chunkers


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
def get_catalog() -> DataCatalog:
    """Load and cache the data-source catalog.

    Read once per process; rolling a new catalog version requires a restart so the
    decisions baked into a running pipeline never change mid-flight.
    """
    return load_catalog(get_settings().CATALOG_PATH)


@lru_cache
def get_filesystem_loader() -> FileSystemLoader:
    return FileSystemLoader(data_root=get_settings().INGESTION_DATA_ROOT)


@lru_cache
def get_parser_registry() -> ParserRegistry:
    return default_registry()


def build_pseudonymizer(session):
    """Build a :class:`ConsistentPseudonymizer` backed by Postgres.

    Not a singleton: the mapping store wraps a Session, so each caller passes its
    own. Imported lazily because Presidio and spaCy are slow to import.
    """
    from app.ingestion.pii import ConsistentPseudonymizer, PostgresMappingStore, build_analyzer

    settings = get_settings()
    return ConsistentPseudonymizer(
        analyzer=build_analyzer(),
        mapping_store=PostgresMappingStore(session),
        salt=settings.PSEUDONYM_HASH_SALT,
        faker_locale=settings.PSEUDONYM_FAKER_LOCALE,
        language="es",
    )


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
