from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def _disable_semantic_cache_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests hit the real endpoint via the ``client`` fixture, which resolves
    ``get_semantic_cache()`` per request. Against a real Redis Stack that cache is
    stateful and persists across tests/runs, so without this, identical
    descriptions across unrelated tests would leak cache hits between them. Tests
    that target the semantic cache itself exercise ``EstimationSemanticCache``
    directly instead of going through this dependency.
    """
    monkeypatch.setattr("app.dependencies.get_semantic_cache", lambda: None)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a FastAPI test client with the lifespan actually running.

    Using ``TestClient(app)`` outside a ``with`` block skips ``lifespan`` entirely
    (Starlette only runs it on ``__enter__``), which previously let a broken
    startup path go undetected. See ``/health``'s docstring for the invariant
    this fixture is meant to exercise: startup must succeed with no API key set.
    """
    with TestClient(app) as test_client:
        yield test_client

