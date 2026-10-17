"""POST /api/v1/ingestion/runs and GET /api/v1/ingestion/jobs/{job_id}.

The router is thin: it validates the source against the catalog, records the job
row, dispatches the orchestrator as a BackgroundTask and returns 202. All the
ingestion logic lives in ``app.ingestion.orchestrator``.

Handlers are plain ``def`` because they block on the database; FastAPI runs them
in its threadpool instead of stalling the event loop.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_catalog, get_filesystem_loader, get_parser_registry
from app.ingestion.catalog.models import CatalogDecision, DataCatalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.orchestrator import ingest_source
from app.ingestion.parsers.registry import ParserRegistry
from app.persistence.database import get_session, new_session
from app.persistence.repositories.jobs import JobsRepository
from app.schemas.ingestion import (
    IngestionJobView,
    IngestionRunRequest,
    IngestionRunResponse,
)

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"])


def _run_in_background(
    *,
    job_id: uuid.UUID,
    source_name: str,
    catalog: DataCatalog,
    loader: FileSystemLoader,
    registry: ParserRegistry,
) -> None:
    """BackgroundTask body. Owns its own Session because the request one is closed."""
    session = new_session()
    try:
        ingest_source(
            catalog=catalog,
            source_name=source_name,
            loader=loader,
            registry=registry,
            jobs_repo=JobsRepository(session),
            job_id=job_id,
        )
    except Exception as exc:  # noqa: BLE001
        # The orchestrator already recorded the failure on the job row.
        log.error(
            "ingestion_background_failed",
            job_id=str(job_id),
            source_name=source_name,
            error=str(exc)[:400],
        )
    finally:
        session.close()


@router.post("/runs", response_model=IngestionRunResponse, status_code=202)
def create_ingestion_run(
    request: IngestionRunRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    catalog: DataCatalog = Depends(get_catalog),
    loader: FileSystemLoader = Depends(get_filesystem_loader),
    registry: ParserRegistry = Depends(get_parser_registry),
) -> IngestionRunResponse:
    source = catalog.find(request.source_name)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail={"reason": "unknown_source", "source_name": request.source_name},
        )
    if source.decision is not CatalogDecision.INCLUDE:
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "source_not_included",
                "source_name": request.source_name,
                "decision": source.decision.value,
                "decision_reason": source.decision_reason,
            },
        )

    job = JobsRepository(session).create(source_name=request.source_name)

    background.add_task(
        _run_in_background,
        job_id=job.job_id,
        source_name=request.source_name,
        catalog=catalog,
        loader=loader,
        registry=registry,
    )
    return IngestionRunResponse(job_id=job.job_id, source_name=job.source_name, status=job.status)


@router.get("/jobs/{job_id}", response_model=IngestionJobView)
def get_ingestion_job(
    job_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> IngestionJobView:
    job = JobsRepository(session).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return IngestionJobView(
        job_id=job.job_id,
        source_name=job.source_name,
        status=job.status,
        documents_count=job.documents_count,
        error_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )
