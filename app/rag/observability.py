"""Per-stage structured logging and request correlation for the RAG pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

import structlog
from starlette.requests import Request

if TYPE_CHECKING:
    from app.rag.schemas import CitationReport

log = structlog.get_logger()


def get_request_id(request: Request) -> str:
    """Correlation id bound by the X-Request-ID middleware, or a fresh UUID if it did not run."""
    return getattr(request.state, "request_id", None) or str(uuid4())


def log_citation_report(report: CitationReport, request_id: str) -> None:
    """Log a citation audit correlated by ``request_id``.

    A dangling citation is a grounding failure, so it is logged at WARNING
    (with the offending ids); a clean report is INFO.
    """
    emit = log.warning if report.has_dangling else log.info
    emit(
        "citation_report",
        request_id=request_id,
        total_lines=report.total_lines,
        grounded_lines=report.grounded_lines,
        dangling_lines=report.dangling_lines,
        insufficient_lines=report.insufficient_lines,
        verified_citations=report.verified_citations,
        dangling_citations=report.dangling_citations,
    )


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
