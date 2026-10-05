"""Semantic cache for the typed estimation endpoint.

Two requests are considered the same when:

1. Their **bucket** matches exactly. The bucket is a deterministic tag composed
   of ``prompt_version:project_type:detail_level:output_format``. Two requests
   with different form options never share a cache entry even if their
   descriptions are similar — the rendered prompt differs, so the estimation
   should differ too.
2. The cosine similarity of their description embeddings is at least
   ``threshold`` (configurable, recommended between 0.85 and 0.92).

When ``log_only=True`` the cache still does the lookup and logs the score, but
never returns a hit — useful for calibrating the threshold against real
traffic before flipping it on in production.

The store uses ``redis/redis-stack``: vanilla ``redis:7-alpine`` lacks the
RediSearch module and ``SearchIndex.create()`` will fail at startup.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from app.schemas.estimation import EstimationRequest, EstimationResult

log = structlog.get_logger()


def _to_bytes(vector: list[float]) -> bytes:
    """RediSearch stores vectors as float32 bytes. redisvl will reject lists."""
    return np.array(vector, dtype=np.float32).tobytes()


_INDEX_SCHEMA: dict[str, Any] = {
    "index": {
        "name": "estimations",
        "prefix": "estimation:semantic",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "bucket", "type": "tag"},
        {"name": "result_json", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 1536,  # text-embedding-3-small
                "distance_metric": "cosine",
                "algorithm": "flat",
            },
        },
    ],
}


class EstimationSemanticCache:
    """Vector-similarity cache on top of redisvl + Redis Stack."""

    def __init__(
        self,
        *,
        redis_client: Any,
        vectorizer: Any,
        threshold: float = 0.85,
        ttl: int = 86400,
        log_only: bool = False,
        index_name: str = "estimations",
    ) -> None:
        from redisvl.index import SearchIndex

        self.redis_client = redis_client
        self.vectorizer = vectorizer
        self.threshold = threshold
        self.ttl = ttl
        self.log_only = log_only

        schema = dict(_INDEX_SCHEMA)
        schema["index"] = {**_INDEX_SCHEMA["index"], "name": index_name}

        self.index = SearchIndex.from_dict(schema)
        self.index.set_client(redis_client)
        try:
            self.index.create(overwrite=False)
        except Exception as exc:  # noqa: BLE001 — already-exists is fine
            log.debug("semantic_index_create_skipped", error=str(exc)[:120])

    # ------------------------------------------------------------------
    # Bucket: the deterministic part of the composite cache key.
    # ------------------------------------------------------------------

    @staticmethod
    def bucket_for(request: EstimationRequest, prompt_version: str) -> str:
        return (
            f"{prompt_version}"
            f":{request.project_type.value}"
            f":{request.detail_level.value}"
            f":{request.output_format.value}"
        )

    # ------------------------------------------------------------------
    # Lookup / store: the vector part of the composite cache key.
    # ------------------------------------------------------------------

    def lookup(self, request: EstimationRequest, prompt_version: str) -> EstimationResult | None:
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        bucket = self.bucket_for(request, prompt_version)
        try:
            embedding = self.vectorizer.embed(request.description)
            query = VectorQuery(
                vector=_to_bytes(embedding),
                vector_field_name="embedding",
                return_fields=["result_json", "bucket"],
                num_results=1,
                return_score=True,
                filter_expression=Tag("bucket") == bucket,
            )
            results = self.index.query(query)
        except Exception as exc:  # noqa: BLE001 — e.g. vanilla Redis without RediSearch
            log.warning("semantic_cache_lookup_failed", bucket=bucket, error=str(exc)[:200])
            return None

        if not results:
            log.info("semantic_cache_miss", bucket=bucket, reason="empty_index")
            return None

        hit = results[0]
        # redisvl returns cosine *distance* (0 = identical, up to 2 = opposite).
        distance = float(hit.get("vector_distance", 1.0))
        similarity = 1.0 - distance
        log.info(
            "semantic_cache_lookup", bucket=bucket,
            similarity=round(similarity, 4), threshold=self.threshold,
        )

        if similarity < self.threshold:
            log.info("semantic_cache_miss", bucket=bucket, reason="below_threshold")
            return None

        if self.log_only:
            log.info("semantic_cache_hit_log_only", bucket=bucket, similarity=round(similarity, 4))
            return None

        log.info("semantic_cache_hit", bucket=bucket, similarity=round(similarity, 4))
        return EstimationResult.model_validate_json(hit["result_json"])

    def store(self, request: EstimationRequest, result: EstimationResult, prompt_version: str) -> None:
        bucket = self.bucket_for(request, prompt_version)
        embedding = self.vectorizer.embed(request.description)
        payload = [
            {
                "bucket": bucket,
                "result_json": result.model_dump_json(),
                "embedding": _to_bytes(embedding),
            }
        ]
        try:
            self.index.load(payload, ttl=self.ttl)
            log.info("semantic_cache_stored", bucket=bucket, ttl=self.ttl)
        except Exception as exc:  # noqa: BLE001 — never block the response on a cache write
            log.warning("semantic_cache_store_failed", error=str(exc)[:200])
