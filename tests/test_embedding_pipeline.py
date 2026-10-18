import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import tiktoken
from openai import RateLimitError
from pydantic import ValidationError

from app.dependencies import get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import BATCH_SIZE, OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import (
    Budget,
    BudgetComponent,
    Chunk,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
    IngestStats,
)
from app.main import app
from scripts import compare
from scripts.compare import cosine_similarity


@pytest.fixture
def budgets():
    path = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"
    return [Budget.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def test_budget_models_preserve_all_sample_fields():
    path = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"
    payload = {"budgets": json.loads(path.read_text(encoding="utf-8"))}
    request = IngestRequest.model_validate(payload)
    assert request.model_dump(mode="json") == payload
    assert all(isinstance(budget, Budget) for budget in request.budgets)
    assert all(
        isinstance(component, BudgetComponent)
        for budget in request.budgets for component in budget.components
    )


@pytest.mark.parametrize(
    "field_path,value,error_type",
    [
        (("client_metadata", "sector"), "unknown", "literal_error"),
        (("components", 0, "complexity"), "extreme", "literal_error"),
        (("components", 0, "estimated_hours"), -1, "greater_than_equal"),
        (("total_estimated_hours",), -1, "greater_than_equal"),
        (("year",), 1999, "greater_than_equal"),
        (("year",), 2101, "less_than_equal"),
        (("components",), [], "too_short"),
    ],
)
def test_budget_rejects_invalid_values(budgets, field_path, value, error_type):
    payload = budgets[0].model_dump()
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    with pytest.raises(ValidationError) as error:
        Budget.model_validate(payload)
    assert any(
        detail["loc"] == field_path and detail["type"] == error_type
        for detail in error.value.errors()
    )


def test_component_defaults_are_independent(budgets):
    payload = budgets[0].components[0].model_dump(exclude={"tech_stack", "dependencies"})
    first = BudgetComponent.model_validate(payload)
    second = BudgetComponent.model_validate(payload)
    first.tech_stack.append("python")
    first.dependencies.append("OTHER-001")
    assert second.tech_stack == []
    assert second.dependencies == []


def test_chunk_metadata_defaults_and_token_validation():
    first = Chunk(chunk_id="first", text="First", token_count=0)
    second = Chunk(chunk_id="second", text="Second", token_count=1)
    first.metadata["year"] = 2024
    assert second.metadata == {}
    with pytest.raises(ValidationError) as error:
        Chunk(chunk_id="invalid", text="Invalid", token_count=-1)
    assert error.value.errors()[0]["loc"] == ("token_count",)


@pytest.mark.parametrize(
    "field", ["total_budgets", "total_chunks", "total_tokens", "estimated_cost_usd"]
)
def test_ingest_stats_reject_negative_values(field):
    payload = dict(total_budgets=1, total_chunks=1, total_tokens=1, estimated_cost_usd=0.0)
    payload[field] = -1
    with pytest.raises(ValidationError) as error:
        IngestStats.model_validate(payload)
    assert error.value.errors()[0]["loc"] == (field,)


def test_ingest_response_validates_vectors_and_serializes_stats():
    payload = {
        "chunks": [{
            "chunk_id": "BUD-2024-001::AUTH-001", "text": "Authentication backend",
            "metadata": {"client_sector": "finance", "year": 2024},
            "token_count": 2, "embedding": [1.0, 0.5],
        }],
        "stats": {
            "total_budgets": 1, "total_chunks": 1, "total_tokens": 2,
            "estimated_cost_usd": estimated_cost_usd(2),
        },
    }
    response = IngestResponse.model_validate(payload)
    assert isinstance(response.chunks[0], EmbeddedChunk)
    assert isinstance(response.chunks[0], Chunk)
    assert isinstance(response.stats, IngestStats)
    assert response.model_dump(mode="json") == payload
    payload["chunks"][0]["embedding"] = ["invalid"]
    with pytest.raises(ValidationError) as error:
        IngestResponse.model_validate(payload)
    assert error.value.errors()[0]["loc"] == ("chunks", 0, "embedding", 0)


def test_chunker_preserves_each_component_and_context(budgets):
    chunks = JSONStructuralChunker().chunk(budgets)
    assert len(chunks) == sum(len(budget.components) for budget in budgets)
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    for chunk, (budget, component) in zip(
        chunks, ((budget, component) for budget in budgets for component in budget.components)
    ):
        assert chunk.chunk_id == f"{budget.budget_id}::{component.component_id}"
        assert chunk.text == (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {budget.client_metadata.sector} | "
            f"Year: {budget.year} | Main tech: {budget.main_technology}]\n\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
        assert chunk.metadata == {
            "budget_id": budget.budget_id,
            "component_id": component.component_id,
            "client_sector": budget.client_metadata.sector,
            "main_technology": budget.main_technology,
            "year": budget.year,
            "complexity": component.complexity,
            "estimated_hours": component.estimated_hours,
        }
        assert chunk.token_count == len(encoding.encode(chunk.text))


def test_chunker_empty_input_returns_no_chunks():
    assert JSONStructuralChunker().chunk([]) == []


def test_chunker_keeps_long_description_in_one_chunk(budgets):
    payload = budgets[0].model_dump()
    component = payload["components"][0]
    component["description"] = "Detailed authentication requirement. " * 5000
    component["tech_stack"] = []
    payload["components"] = [component]
    budget = Budget.model_validate(payload)

    chunks = JSONStructuralChunker().chunk([budget])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == f"{budget.budget_id}::{component['component_id']}"
    assert chunk.text.count(component["description"]) == 1
    assert "Tech stack: \n" in chunk.text
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    assert chunk.token_count == len(encoding.encode(chunk.text))
    assert chunk.token_count > 8191


def test_embedder_batches_and_preserves_order():
    client = Mock()
    client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[float(text)]) for text in kwargs["input"]]
    )
    chunks = [
        Chunk(chunk_id=str(index), text=str(index), token_count=1)
        for index in range(BATCH_SIZE + 1)
    ]
    embedded = OpenAIEmbedder(client).embed_many(chunks)
    assert [chunk.embedding for chunk in embedded] == [[float(index)] for index in range(101)]
    assert [chunk.chunk_id for chunk in embedded] == [chunk.chunk_id for chunk in chunks]
    assert [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list] == [100, 1]
    assert all(
        call.kwargs["model"] == "text-embedding-3-small"
        for call in client.embeddings.create.call_args_list
    )


def test_embedder_empty_input_does_not_call_provider():
    client = Mock()
    assert OpenAIEmbedder(client).embed_many([]) == []
    client.embeddings.create.assert_not_called()


def test_embedder_one_keeps_default_dimensions():
    client = Mock()
    vector = [0.5] * 1536
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=vector)]
    )

    assert OpenAIEmbedder(client).embed_one("text") == vector
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["text"]
    )


def test_embedder_logs_each_batch(monkeypatch):
    client = Mock()
    client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.5]) for text in kwargs["input"]]
    )
    chunks = [
        Chunk(chunk_id=str(index), text=str(index), token_count=2)
        for index in range(BATCH_SIZE + 1)
    ]
    logger = Mock()
    monkeypatch.setattr("app.embedding_pipeline.embedder.log", logger)
    monkeypatch.setattr(
        "app.embedding_pipeline.embedder.time.perf_counter",
        Mock(side_effect=[0.0, 0.125, 1.0, 1.25]),
    )

    OpenAIEmbedder(client).embed_many(chunks)

    assert [call.args for call in logger.info.call_args_list] == [
        ("embedding_batch_done",), ("embedding_batch_done",)
    ]
    assert [call.kwargs for call in logger.info.call_args_list] == [
        {"chunks": 100, "tokens": 200, "latency_ms": 125.0,
         "model": "text-embedding-3-small"},
        {"chunks": 1, "tokens": 2, "latency_ms": 250.0,
         "model": "text-embedding-3-small"},
    ]


def test_embedder_propagates_other_errors_without_retry(monkeypatch):
    client = Mock()
    error = RuntimeError("provider unavailable")
    client.embeddings.create.side_effect = error
    sleep = Mock()
    monkeypatch.setattr("app.embedding_pipeline.embedder.time.sleep", sleep)

    with pytest.raises(RuntimeError) as raised:
        OpenAIEmbedder(client).embed_one("text")

    assert raised.value is error
    client.embeddings.create.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("exhausted", [False, True])
def test_embedder_retries_rate_limits(monkeypatch, exhausted):
    client = Mock()
    error = RateLimitError(
        "rate limit", response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")),
        body=None,
    )
    sleep = Mock()
    monkeypatch.setattr("app.embedding_pipeline.embedder.time.sleep", sleep)
    logger = Mock()
    monkeypatch.setattr("app.embedding_pipeline.embedder.log", logger)
    if exhausted:
        client.embeddings.create.side_effect = error
        with pytest.raises(RateLimitError) as raised:
            OpenAIEmbedder(client).embed_one("text")
        assert raised.value is error
        assert [call.args[0] for call in sleep.call_args_list] == [1, 2, 4]
        assert client.embeddings.create.call_count == 4
        assert [call.kwargs["retry_in_s"] for call in logger.warning.call_args_list[:-1]] == [
            1, 2, 4,
        ]
        logger.warning.assert_called_with("embedding_rate_limit_exhausted", attempts=4)
    else:
        client.embeddings.create.side_effect = [
            error, SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])
        ]
        assert OpenAIEmbedder(client).embed_one("text") == [1.0, 0.0]
        sleep.assert_called_once_with(1)


def test_ingest_returns_vectors_and_stats(client, budgets):
    api_client = Mock()
    api_client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 0.0]) for text in kwargs["input"]]
    )
    app.dependency_overrides[get_embedder] = lambda: OpenAIEmbedder(api_client)
    try:
        response = client.post("/embeddings/ingest", json={"budgets": [b.model_dump() for b in budgets]})
    finally:
        app.dependency_overrides.pop(get_embedder, None)
    assert response.status_code == 200
    body = response.json()
    chunks = JSONStructuralChunker().chunk(budgets)
    tokens = sum(chunk.token_count for chunk in chunks)
    assert body["stats"] == {
        "total_budgets": len(budgets), "total_chunks": len(chunks),
        "total_tokens": tokens, "estimated_cost_usd": estimated_cost_usd(tokens),
    }
    assert body["stats"]["estimated_cost_usd"] == pytest.approx(tokens * 0.02 / 1_000_000)
    assert body["chunks"] == [dict(chunk.model_dump(), embedding=[1.0, 0.0]) for chunk in chunks]


@pytest.mark.parametrize("unavailable", [False, True])
def test_ingest_errors_are_generic(client, budgets, unavailable, monkeypatch):
    logger = Mock()
    monkeypatch.setattr("app.embedding_pipeline.router.log", logger)
    embedder = Mock()
    embedder.embed_many.side_effect = RuntimeError("sk-proj-private-detail")
    app.dependency_overrides[get_embedder] = lambda: None if unavailable else embedder
    try:
        response = client.post("/embeddings/ingest", json={"budgets": [budgets[0].model_dump()]})
    finally:
        app.dependency_overrides.pop(get_embedder, None)
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Embedding service is not available." if unavailable else "Failed to generate embeddings."
    )
    assert "sk-proj" not in response.text
    if unavailable:
        logger.error.assert_called_once_with(
            "embeddings_ingest_failed", reason="embedder_unavailable"
        )
    else:
        logger.error.assert_called_once_with(
            "embeddings_ingest_failed", reason="embedding_api_error",
            error_type="RuntimeError", error="sk-proj-private-detail",
        )


def test_ingest_is_documented_with_executable_sample(client, budgets):
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "/openapi.json" in docs.text
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/embeddings/ingest"]["post"]
    assert operation["tags"] == ["embeddings"]
    assert set(operation["responses"]) == {"200", "422", "500"}
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/IngestRequest"
    }
    example = schema["components"]["schemas"]["IngestRequest"]["examples"][0]
    assert example == {"budgets": [budgets[0].model_dump(mode="json")]}
    api_client = Mock()
    api_client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 0.0]) for text in kwargs["input"]]
    )
    app.dependency_overrides[get_embedder] = lambda: OpenAIEmbedder(api_client)
    try:
        response = client.post("/embeddings/ingest", json=example)
    finally:
        app.dependency_overrides.pop(get_embedder, None)
    assert response.status_code == 200
    result = IngestResponse.model_validate(response.json())
    assert result.stats.total_budgets == 1
    assert result.stats.total_chunks == len(budgets[0].components)
    assert all(chunk.embedding == [1.0, 0.0] for chunk in result.chunks)


def test_ingest_rejects_empty_budgets(client):
    embedder = Mock()
    app.dependency_overrides[get_embedder] = lambda: embedder
    try:
        response = client.post("/embeddings/ingest", json={"budgets": []})
    finally:
        app.dependency_overrides.pop(get_embedder, None)
    assert response.status_code == 422
    embedder.embed_many.assert_not_called()


@pytest.mark.parametrize(
    "vector_a,vector_b,expected",
    [([1.0, 0.0], [1.0, 0.0], 1.0), ([1.0, 0.0], [0.0, 1.0], 0.0),
     ([1.0, 0.0], [-1.0, 0.0], -1.0), ([0.0, 0.0], [1.0, 0.0], 0.0),
     ([3.0, 4.0], [4.0, 3.0], 0.96)],
)
def test_cosine_similarity(vector_a, vector_b, expected):
    assert cosine_similarity(vector_a, vector_b) == pytest.approx(expected)


def test_compare_cli(monkeypatch, capsys):
    settings = SimpleNamespace(OPENAI_API_KEY="test-key", EMBEDDING_MODEL="text-embedding-3-small")
    monkeypatch.setattr(compare, "get_settings", lambda: settings)
    api_client = Mock()
    api_client.embeddings.create.side_effect = [
        SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0])]),
        SimpleNamespace(data=[SimpleNamespace(embedding=[4.0, 3.0])]),
    ]
    client_factory = Mock(return_value=api_client)
    monkeypatch.setattr(compare, "OpenAI", client_factory)
    monkeypatch.setattr("sys.argv", ["compare.py", "--text-a", "OAuth backend", "--text-b", "JWT service"])

    assert compare.main() == 0

    client_factory.assert_called_once_with(api_key="test-key")
    assert [call.kwargs for call in api_client.embeddings.create.call_args_list] == [
        {"model": settings.EMBEDDING_MODEL, "input": ["OAuth backend"]},
        {"model": settings.EMBEDDING_MODEL, "input": ["JWT service"]},
    ]
    captured = capsys.readouterr()
    assert captured.out == (
        "Text A: OAuth backend\n"
        "Text B: JWT service\n"
        "Cosine similarity: 0.9600\n"
    )
    assert captured.err == ""


def test_compare_cli_requires_openai_key(monkeypatch, capsys):
    monkeypatch.setattr(compare, "get_settings", lambda: SimpleNamespace(OPENAI_API_KEY=None))
    client_factory = Mock()
    monkeypatch.setattr(compare, "OpenAI", client_factory)
    monkeypatch.setattr("sys.argv", ["compare.py", "--text-a", "A", "--text-b", "B"])

    assert compare.main() == 1

    client_factory.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR: OPENAI_API_KEY is not set (check your .env).\n"


@pytest.mark.parametrize("arguments", [[], ["--text-a", "A"], ["--text-b", "B"]])
def test_compare_cli_requires_both_texts(monkeypatch, arguments, capsys):
    monkeypatch.setattr("sys.argv", ["compare.py", *arguments])
    with pytest.raises(SystemExit) as error:
        compare.main()
    assert error.value.code == 2
    assert "required" in capsys.readouterr().err


def test_estimated_cost():
    assert estimated_cost_usd(1_000_000) == pytest.approx(0.02)
    assert estimated_cost_usd(0) == 0.0