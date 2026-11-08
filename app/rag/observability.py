"""Per-stage structured logging and request correlation for the RAG pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4

import structlog
from starlette.requests import Request

log = structlog.get_logger()


def get_request_id(request: Request) -> str:
    """Correlation id bound by the X-Request-ID middleware, or a fresh UUID if it did not run."""
    return getattr(request.state, "request_id", None) or str(uuid4())


@contextmanager
def log_stage(stage: str, request_id: str, **context: Any) -> Iterator[None]:
    """Emit ``stage.started`` / ``stage.completed`` / ``stage.failed`` for one pipeline step."""
    log.info("stage.started", stage=stage, request_id=request_id, **context)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        log.error(
            "stage.failed",
            stage=stage,
            request_id=request_id,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            **context,
        )
        raise
    log.info(
        "stage.completed",
        stage=stage,
        request_id=request_id,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        **context,
    )
