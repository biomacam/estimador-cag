"""Unit tests for the semantic cache's composite key (bucket + vector) and its
fail-open behaviour when RediSearch is unavailable (e.g. vanilla Redis).

All Redis/redisvl internals are faked: these tests must stay fast and never
require a real Redis Stack instance or an OpenAI embeddings call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.cache.semantic import EstimationSemanticCache
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    ProjectType,
)


def _request(
    *,
    project_type: ProjectType = ProjectType.WEB_SAAS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.LINE_ITEMS,
    description: str = "Build a booking system for a yoga studio with class scheduling.",
) -> EstimationRequest:
    return EstimationRequest(
        description=description, project_type=project_type,
        detail_level=detail_level, output_format=output_format,
    )


def _result() -> EstimationResult:
    return EstimationResult(
        summary="A small SaaS with authentication and an admin dashboard.",
        confidence_pct=80,
        phases=[{
            "name": "Implementation", "duration_weeks": 4, "cost_eur": 10000,
            "summary": "Build and integrate the core SaaS features.",
        }],
        total_duration_weeks=4,
        total_cost_eur=10000,
    )


@pytest.fixture
def cache() -> EstimationSemanticCache:
    with patch("redisvl.index.SearchIndex.from_dict") as from_dict:
        index = MagicMock()
        from_dict.return_value = index
        instance = EstimationSemanticCache(
            redis_client=MagicMock(), vectorizer=MagicMock(), threshold=0.85,
        )
    instance.index = index
    return instance


def test_bucket_is_deterministic_and_excludes_the_description() -> None:
    bucket_a = EstimationSemanticCache.bucket_for(
        _request(description="A yoga studio booking project with scheduling."), prompt_version="v1",
    )
    bucket_b = EstimationSemanticCache.bucket_for(
        _request(description="Totally different project involving a delivery logistics app."),
        prompt_version="v1",
    )
    assert bucket_a == bucket_b == "v1:web_saas:medium:line_items"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_type", ProjectType.MOBILE_APP),
        ("detail_level", DetailLevel.DETAILED),
        ("output_format", OutputFormat.NARRATIVE),
    ],
)
def test_bucket_changes_when_any_form_option_changes(field: str, value: object) -> None:
    base_bucket = EstimationSemanticCache.bucket_for(_request(), prompt_version="v1")
    changed_bucket = EstimationSemanticCache.bucket_for(
        _request(**{field: value}), prompt_version="v1",
    )
    assert changed_bucket != base_bucket


def test_bucket_changes_with_prompt_version() -> None:
    assert (
        EstimationSemanticCache.bucket_for(_request(), prompt_version="v1")
        != EstimationSemanticCache.bucket_for(_request(), prompt_version="v2")
    )


def test_lookup_filters_by_bucket_tag(cache: EstimationSemanticCache) -> None:
    cache.vectorizer.embed.return_value = [0.1] * 1536
    cache.index.query.return_value = []

    cache.lookup(_request(), prompt_version="v1")

    query = cache.index.query.call_args.args[0]
    assert str(query.filter) == '@bucket:{v1\\:web_saas\\:medium\\:line_items}'


def test_similarity_at_or_above_threshold_is_a_hit(cache: EstimationSemanticCache) -> None:
    cache.vectorizer.embed.return_value = [0.1] * 1536
    cache.index.query.return_value = [
        {"result_json": _result().model_dump_json(), "vector_distance": 0.1}  # similarity 0.9
    ]

    hit = cache.lookup(_request(), prompt_version="v1")

    assert hit == _result()


def test_similarity_below_threshold_is_a_miss(cache: EstimationSemanticCache) -> None:
    cache.vectorizer.embed.return_value = [0.1] * 1536
    cache.index.query.return_value = [
        {"result_json": _result().model_dump_json(), "vector_distance": 0.3}  # similarity 0.7
    ]

    assert cache.lookup(_request(), prompt_version="v1") is None


def test_log_only_mode_never_returns_a_hit(cache: EstimationSemanticCache) -> None:
    cache.log_only = True
    cache.vectorizer.embed.return_value = [0.1] * 1536
    cache.index.query.return_value = [
        {"result_json": _result().model_dump_json(), "vector_distance": 0.0}
    ]

    assert cache.lookup(_request(), prompt_version="v1") is None


def test_lookup_fails_open_when_redisearch_is_unavailable(cache: EstimationSemanticCache) -> None:
    """Vanilla Redis lacks RediSearch: a query error must degrade to a miss,
    never bubble up and break the whole /estimate request."""
    cache.vectorizer.embed.return_value = [0.1] * 1536
    cache.index.query.side_effect = Exception("unknown command 'FT.SEARCH'")

    assert cache.lookup(_request(), prompt_version="v1") is None


def test_store_fails_open_when_redisearch_is_unavailable(cache: EstimationSemanticCache) -> None:
    cache.vectorizer.embed.return_value = [0.1] * 1536
    cache.index.load.side_effect = Exception("unknown command 'FT.CREATE'")

    cache.store(_request(), _result(), prompt_version="v1")  # must not raise


def test_store_writes_the_bucket_and_embedding(cache: EstimationSemanticCache) -> None:
    cache.vectorizer.embed.return_value = [0.1] * 1536

    cache.store(_request(), _result(), prompt_version="v1")

    payload = cache.index.load.call_args.args[0]
    assert payload[0]["bucket"] == "v1:web_saas:medium:line_items"
    assert payload[0]["result_json"] == _result().model_dump_json()
