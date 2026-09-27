from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


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
