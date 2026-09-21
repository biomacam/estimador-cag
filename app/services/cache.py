"""Exact-match Redis cache for LLM responses.

The cache key is a SHA-256 of the *full* system prompt plus the user message
plus the generation knobs (model, max_tokens, thinking_budget). That means any
change in Session 2 controls (preprocessing, num_examples, example_format,
ACTIVE_OUTPUT_PROMPT) implicitly invalidates the cache without manual flushing,
because those changes alter the system prompt text.
"""
#Hace que Python trate las anotaciones de tipos de forma diferida.
#Gracias a esto puede escribir:
#def from_url(...) -> "EstimationCache":
#Aunque la propia clase EstimationCache todavía se esté definiendo

from __future__ import annotations
#Se utiliza para calcular el hash SHA-256 con el contenido del prompt, knobs y demás información relevante para  la caché.
import hashlib
import json
from typing import Any

#Es el cliente Python redis-py.Se utiliza para:
# - Crear la conexión.
# - Leer claves.
# - Guardar claves con caducidad.
# - Capturar errores de Redis.

import redis
import structlog

log = structlog.get_logger()


class EstimationCache:
    """Thin wrapper around redis-py with deterministic keying and TTL."""
    #constructor con el tiempo que se conservará la respuesta(TTL) y conexión a Redis.
    #inicializa la instancia
    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl
    #gestiona la clase, permitiendo crear una instancia a partir de una URL de Redis.
    @classmethod
    def from_url(cls, url: str, ttl: int = 86400) -> "EstimationCache":
        return cls(redis.from_url(url, decode_responses=True), ttl=ttl)
    #genera la clave de caché a partir del prompt del sistema, mensaje del usuario y parámetros de generación.
    @staticmethod
    def make_key(
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        thinking_budget: int | None,
    ) -> str:
        payload = json.dumps(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model": model,
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"estimation:{digest}"

    #recupera la respuesta de la caché utilizando la clave generada.
    def get(self, key: str) -> dict[str, Any] | None:
        try:
            cached = self.redis.get(key)
        except redis.RedisError as exc:
            log.warning("cache_get_failed", error=str(exc))
            return None
        if cached:
            log.info("cache_hit", key_prefix=key[:24])
            return json.loads(cached)
        log.info("cache_miss", key_prefix=key[:24])
        return None

    #almacena la respuesta en la caché con la clave generada y el TTL configurado.
    def set(self, key: str, response: dict[str, Any]) -> None:
        try:
            self.redis.setex(key, self.ttl, json.dumps(response))
            log.info("cache_stored", key_prefix=key[:24], ttl=self.ttl)
        except redis.RedisError as exc:
            log.warning("cache_set_failed", error=str(exc))
