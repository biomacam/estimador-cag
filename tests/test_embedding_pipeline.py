import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import tiktoken
from openai import RateLimitError
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.dependencies import get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import BATCH_SIZE, OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import (
    Budget,
    BudgetComponent,
    Chunk,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from app.persistence.database import get_async_session
from app.persistence.models import ChunkRow, DocumentRow
from app.main import app
from app.embedding_pipeline.similarity import cosine_similarity
from scripts import query_examples


@pytest.fixture
def budgets():
    path = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"
    return [Budget.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


class _FakeTransaction:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, exception_type, exception, traceback):
        self.session.committed = exception_type is None
        self.session.rolled_back = exception_type is not None


class _FakeAsyncSession:
    def __init__(self, existing_id=None, search_rows=None):
        self.existing_id = existing_id
        self.search_rows = search_rows or []
        self.executed_statement = None
        self.added = []
        self.added_chunks = []
        self.committed = False
        self.rolled_back = False

    def begin(self):
        return _FakeTransaction(self)

    async def scalar(self, statement):
        return self.existing_id

    def add(self, value):
        self.added.append(value)
        if isinstance(value, DocumentRow):
            value.id = 42

    async def flush(self):
        pass

    def add_all(self, values):
        self.added_chunks.extend(values)

    async def execute(self, statement):
        self.executed_statement = statement
        return _FakeSearchResult(self.search_rows)


class _FakeSearchResult:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


async def _yield_session(session):
    yield session


def test_budget_models_preserve_all_sample_fields():
    path = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"
    budget = json.loads(path.read_text(encoding="utf-8"))[0]
    payload = {
        "source_path": "data/budgets/budget_2024_q1_fintech.json",
        "document_type": "historical_budget",
        "content": budget,
    }
    request = IngestRequest.model_validate(payload)
    assert request.model_dump(mode="json") == payload
    assert isinstance(request.content, Budget)
    assert all(
        isinstance(component, BudgetComponent)
        for component in request.content.components
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


def test_ingest_response_validates_and_serializes_metrics():
    payload = {
        "document_id": 42,
        "chunks_created": 17,
        "embedding_dimension": 1536,
        "ingestion_time_ms": 1240,
    }
    response = IngestResponse.model_validate(payload)
    assert response.model_dump(mode="json") == payload
    payload["chunks_created"] = -1
    with pytest.raises(ValidationError) as error:
        IngestResponse.model_validate(payload)
    assert error.value.errors()[0]["loc"] == ("chunks_created",)


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


def test_embedder_can_send_all_chunks_in_one_request():
    client = Mock()
    client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[float(text)]) for text in kwargs["input"]]
    )
    chunks = [
        Chunk(chunk_id=str(index), text=str(index), token_count=1)
        for index in range(BATCH_SIZE + 1)
    ]

    embedded = OpenAIEmbedder(client).embed_all(chunks)

    assert len(embedded) == len(chunks)
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=[chunk.text for chunk in chunks]
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


def _ingest_payload(budget):
    return {
        "source_path": "data/budgets/budget_2024_q1_fintech.json",
        "document_type": "historical_budget",
        "content": budget.model_dump(mode="json"),
    }


def _install_ingest_overrides(session, embedder):
    async def override_session():
        yield session

    app.dependency_overrides[get_async_session] = override_session
    app.dependency_overrides[get_embedder] = lambda: embedder


def _remove_ingest_overrides():
    app.dependency_overrides.pop(get_async_session, None)
    app.dependency_overrides.pop(get_embedder, None)


def test_ingest_persists_document_and_chunks_with_one_embedding_call(client, budgets):
    api_client = Mock()
    vector = [1.0] * 1536
    api_client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=vector) for text in kwargs["input"]]
    )
    session = _FakeAsyncSession()
    _install_ingest_overrides(session, OpenAIEmbedder(api_client))
    try:
        response = client.post("/embeddings/ingest", json=_ingest_payload(budgets[0]))
    finally:
        _remove_ingest_overrides()

    assert response.status_code == 200
    body = response.json()
    chunks = JSONStructuralChunker().chunk([budgets[0]])
    assert body["document_id"] == 42
    assert body["chunks_created"] == len(chunks)
    assert body["embedding_dimension"] == 1536
    assert body["ingestion_time_ms"] >= 0
    assert "chunks" not in body
    api_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=[chunk.text for chunk in chunks]
    )
    assert session.committed and not session.rolled_back
    assert len(session.added) == 1
    document = session.added[0]
    assert isinstance(document, DocumentRow)
    assert document.source_path == "data/budgets/budget_2024_q1_fintech.json"
    assert document.document_type == "historical_budget"
    assert len(session.added_chunks) == len(chunks)
    assert all(isinstance(row, ChunkRow) for row in session.added_chunks)
    assert all(
        row.document_id == 42
        and row.chunk_type == "budget_component"
        and row.embedding == vector
        for row in session.added_chunks
    )


def test_ingest_duplicate_source_path_returns_existing_document_id(client, budgets):
    api_client = Mock()
    session = _FakeAsyncSession(existing_id=42)
    _install_ingest_overrides(session, OpenAIEmbedder(api_client))
    try:
        response = client.post("/embeddings/ingest", json=_ingest_payload(budgets[0]))
    finally:
        _remove_ingest_overrides()

    assert response.status_code == 409
    assert response.json() == {"detail": "Document already ingested", "document_id": 42}
    assert session.added == []
    api_client.embeddings.create.assert_not_called()


@pytest.mark.parametrize("unavailable", [False, True])
def test_ingest_errors_are_generic(client, budgets, unavailable, monkeypatch):
    logger = Mock()
    monkeypatch.setattr("app.embedding_pipeline.router.log", logger)
    session = _FakeAsyncSession()
    embedder = Mock()
    embedder.embed_all.side_effect = RuntimeError("sk-proj-private-detail")
    _install_ingest_overrides(session, None if unavailable else embedder)
    try:
        response = client.post("/embeddings/ingest", json=_ingest_payload(budgets[0]))
    finally:
        _remove_ingest_overrides()
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Embedding service is not available." if unavailable else "Failed to ingest document."
    )
    assert "sk-proj" not in response.text
    if unavailable:
        assert not session.added
        logger.error.assert_called_once_with(
            "embeddings_ingest_failed", reason="embedder_unavailable"
        )
    else:
        assert session.rolled_back
        logger.error.assert_called_once_with(
            "embeddings_ingest_failed", reason="ingestion_error",
            error_type="RuntimeError", error="sk-proj-private-detail",
        )


def test_ingest_is_documented_with_executable_sample(client, budgets):
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "/openapi.json" in docs.text
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/embeddings/ingest"]["post"]
    assert operation["tags"] == ["embeddings"]
    assert set(operation["responses"]) == {"200", "409", "422", "500"}
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/IngestRequest"
    }
    example = schema["components"]["schemas"]["IngestRequest"]["examples"][0]
    assert example["source_path"] == "data/budgets/budget_2024_q1_fintech.json"
    assert example["document_type"] == "historical_budget"
    assert isinstance(example["content"], dict)
    api_client = Mock()
    api_client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0] * 1536) for text in kwargs["input"]]
    )
    session = _FakeAsyncSession()
    _install_ingest_overrides(session, OpenAIEmbedder(api_client))
    try:
        response = client.post("/embeddings/ingest", json=example)
    finally:
        _remove_ingest_overrides()
    assert response.status_code == 200
    result = IngestResponse.model_validate(response.json())
    assert result.chunks_created == len(example["content"]["components"])
    assert result.embedding_dimension == 1536
    api_client.embeddings.create.assert_called_once()


def test_ingest_rejects_empty_source_path(client):
    embedder = Mock()
    session = _FakeAsyncSession()
    _install_ingest_overrides(session, embedder)
    payload = {
        "source_path": "",
        "document_type": "historical_budget",
        "content": {"budget_id": "invalid"},
    }
    try:
        response = client.post("/embeddings/ingest", json=payload)
    finally:
        _remove_ingest_overrides()
    assert response.status_code == 422
    embedder.embed_all.assert_not_called()
    assert not session.added


def test_search_embeds_query_and_returns_cosine_nearest_chunks(client):
    query = "REST API with OAuth authentication for fintech sector"
    rows = [{
        "chunk_id": 156,
        "document_id": 12,
        "chunk_type": "structural",
        "content": "Backend service implementation with JWT-based authentication...",
        "distance": 0.231,
        "metadata": {"scope": "backend", "technologies": ["python", "fastapi"]},
    }]
    session = _FakeAsyncSession(search_rows=rows)
    api_client = Mock()
    api_client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.25] * 1536)]
    )
    _install_ingest_overrides(session, OpenAIEmbedder(api_client))
    try:
        response = client.post("/search", json={"query": query, "k": 5})
    finally:
        _remove_ingest_overrides()

    assert response.status_code == 200
    assert response.json() == {
        "query": query,
        "k": 5,
        "search_time_ms": response.json()["search_time_ms"],
        "results": rows,
    }
    assert response.json()["search_time_ms"] >= 0
    api_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=[query]
    )
    sql = str(session.executed_statement.compile(dialect=postgresql.dialect()))
    assert " <=> " in sql
    assert "ORDER BY" in sql
    assert "LIMIT" in sql


def test_search_request_requires_positive_k():
    with pytest.raises(ValidationError):
        SearchRequest(query="OAuth API", k=0)


def test_search_response_matches_contract():
    payload = {
        "query": "OAuth API",
        "k": 1,
        "search_time_ms": 87,
        "results": [{
            "chunk_id": 156,
            "document_id": 12,
            "chunk_type": "budget_component",
            "content": "Backend service implementation",
            "distance": 0.231,
            "metadata": {"scope": "backend", "technologies": ["python"]},
        }],
    }
    assert SearchResponse.model_validate(payload).model_dump() == payload


@pytest.mark.parametrize(
    "vector_a,vector_b,expected",
    [([1.0, 0.0], [1.0, 0.0], 1.0), ([1.0, 0.0], [0.0, 1.0], 0.0),
     ([1.0, 0.0], [-1.0, 0.0], -1.0), ([0.0, 0.0], [1.0, 0.0], 0.0),
     ([3.0, 4.0], [4.0, 3.0], 0.96)],
)
def test_cosine_similarity(vector_a, vector_b, expected):
    assert cosine_similarity(vector_a, vector_b) == pytest.approx(expected)


class _FakeSearchResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _FakeSearchClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.health_checks = []
        self.calls = []
        self.ingest_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        pass

    def get(self, url):
        self.health_checks.append(url)
        return _FakeSearchResponse()

    def post(self, path, json):
        self.calls.append((path, json))
        if path.endswith("/embeddings/ingest"):
            self.ingest_calls += 1
            status_code = 200 if self.ingest_calls == 1 else 409
            return _FakeSearchResponse(status_code=status_code)

        return _FakeSearchResponse({
            "query": json["query"],
            "k": json["k"],
            "search_time_ms": 87,
            "results": [{
                "chunk_id": 156,
                "document_id": 12,
                "chunk_type": "budget_component",
                "content": "Backend service implementation " * 8,
                "distance": 0.231,
                "metadata": {},
            }],
        })


def test_query_examples_calls_search_and_formats_top_results(monkeypatch, capsys):
    settings = SimpleNamespace(ESTIMATOR_API_BASE_URL="http://localhost:8000")
    monkeypatch.setattr(query_examples, "get_settings", lambda: settings)
    clients = []

    def client_factory(**kwargs):
        client = _FakeSearchClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(query_examples.httpx, "Client", client_factory)

    assert query_examples.main() == 0

    client = clients[0]
    budget_count = len(json.loads(query_examples.CORPUS_PATH.read_text(encoding="utf-8")))
    assert client.kwargs == {"timeout": 120.0}
    assert client.health_checks == ["http://localhost:8000/health"]
    ingest_calls = [call for call in client.calls if call[0].endswith("/embeddings/ingest")]
    search_calls = [call for call in client.calls if call[0].endswith("/search")]
    assert len(ingest_calls) == budget_count
    assert all(path.startswith("http://localhost:8000/") for path, _ in client.calls)
    assert all(body["k"] == 5 for _, body in search_calls)
    assert [body["query"] for _, body in search_calls] == [query for _, query in query_examples.QUERIES]
    assert ingest_calls[0][1]["source_path"].startswith("data/budgets_sample.json::")
    output = capsys.readouterr().out
    assert f"Corpus: {budget_count} budgets - 1 ingested, {budget_count - 1} already present." in output
    result_lines = [line for line in output.splitlines() if "0.2310" in line]
    assert len(result_lines) == 5
    assert all("budget_component" in line for line in result_lines)
    previews = [line.rsplit("  ", 1)[-1] for line in result_lines]
    assert all(len(preview) == query_examples.CONTENT_PREVIEW_CHARS for preview in previews)


def test_query_examples_reports_api_connection_errors(monkeypatch, capsys):
    class FailingClient(_FakeSearchClient):
        def get(self, url):
            request = httpx.Request("GET", url)
            raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(
        query_examples,
        "get_settings",
        lambda: SimpleNamespace(ESTIMATOR_API_BASE_URL="http://localhost:8000"),
    )
    monkeypatch.setattr(query_examples.httpx, "Client", FailingClient)

    with pytest.raises(SystemExit) as error:
        query_examples.main()
    assert error.value.code == 1
    assert "ERROR: no estimator API reachable." in capsys.readouterr().err


def test_estimated_cost():
    assert estimated_cost_usd(1_000_000) == pytest.approx(0.02)
    assert estimated_cost_usd(0) == 0.0