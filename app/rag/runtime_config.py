"""Runtime-mutable retrieval toggles (search mode + reranking), backed by Redis.

``Settings`` (.env) holds the immutable defaults; overrides live in a Redis hash so
the search mode and reranking can be flipped mid-session without a restart, and
are read on every call. Reads degrade to the settings default if Redis is down;
writes re-raise ``RuntimeConfigUnavailable`` so a failed override is visible (503).
"""

from __future__ import annotations

import redis
import structlog

from app.config import Settings

log = structlog.get_logger()

RETRIEVAL_HASH_KEY = "estimator:runtime_retrieval"

SEARCH_MODE_KEY = "RETRIEVAL_SEARCH_MODE"
RERANK_KEY = "RERANKER_ENABLED"

_VALID_SEARCH_MODES = ("vector", "hybrid")


class RuntimeConfigUnavailable(Exception):
    """Raised when an override WRITE fails (Redis unreachable)."""


class RuntimeRetrievalConfig:
    """Redis-hash-backed override store for the retrieval toggles."""

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> "RuntimeRetrievalConfig":
        return cls(redis.from_url(url, decode_responses=True), settings)

    def _get_raw(self, key: str) -> str | None:
        try:
            return self._redis.hget(RETRIEVAL_HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key=key, error=str(exc)[:200])
            return None

    def _set_raw(self, key: str, value: str | None) -> None:
        try:
            if value is None:
                self._redis.hdel(RETRIEVAL_HASH_KEY, key)
            else:
                self._redis.hset(RETRIEVAL_HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective_search_mode(self) -> str:
        override = self._get_raw(SEARCH_MODE_KEY)
        if override in _VALID_SEARCH_MODES:
            return override
        return self._settings.RETRIEVAL_SEARCH_MODE

    def effective_rerank(self) -> bool:
        override = self._get_raw(RERANK_KEY)
        if override is None:
            return self._settings.RERANKER_ENABLED
        return override.lower() == "true"

    def set_search_mode(self, value: str | None) -> None:
        """``None`` clears the override (back to the .env default)."""
        if value is not None and value not in _VALID_SEARCH_MODES:
            raise ValueError(f"Invalid search mode: {value}")
        self._set_raw(SEARCH_MODE_KEY, value)

    def set_rerank(self, value: bool | None) -> None:
        self._set_raw(RERANK_KEY, None if value is None else str(value).lower())

    def snapshot(self) -> dict[str, dict[str, object]]:
        """``{effective, default, overridden}`` per toggle."""
        try:
            overrides = self._redis.hgetall(RETRIEVAL_HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            SEARCH_MODE_KEY: {
                "effective": self.effective_search_mode(),
                "default": self._settings.RETRIEVAL_SEARCH_MODE,
                "overridden": SEARCH_MODE_KEY in overrides,
            },
            RERANK_KEY: {
                "effective": self.effective_rerank(),
                "default": self._settings.RERANKER_ENABLED,
                "overridden": RERANK_KEY in overrides,
            },
        }

    def reset_all(self) -> None:
        """Drop every override (back to .env defaults)."""
        try:
            self._redis.delete(RETRIEVAL_HASH_KEY)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc
