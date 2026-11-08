"""``/api/v1/config/retrieval`` — flip search mode and reranking at runtime."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.dependencies import get_runtime_retrieval_config
from app.rag.runtime_config import RuntimeConfigUnavailable, RuntimeRetrievalConfig

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/config", tags=["config"])


class RetrievalUpdateRequest(BaseModel):
    """Partial update: only the keys present are touched; ``null`` resets a knob to its default."""

    search_mode: str | None = Field(default=None, description="'vector' or 'hybrid'.")
    rerank: bool | None = Field(default=None, description="Enable cross-encoder reranking.")


def _payload(runtime_retrieval: RuntimeRetrievalConfig, settings: Settings) -> dict:
    return {
        "retrieval": runtime_retrieval.snapshot(),
        "reranker_model": settings.RERANKER_MODEL,
    }


@router.get("/retrieval")
def get_retrieval(
    runtime_retrieval: RuntimeRetrievalConfig = Depends(get_runtime_retrieval_config),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Effective / default / overridden value per toggle."""
    return _payload(runtime_retrieval, settings)


@router.put("/retrieval")
def update_retrieval(
    request: RetrievalUpdateRequest,
    runtime_retrieval: RuntimeRetrievalConfig = Depends(get_runtime_retrieval_config),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Effective on the next retrieval, no restart. ``model_fields_set`` separates
    "reset to default" (explicit ``null``) from "leave untouched" (key absent)."""
    sent = request.model_fields_set
    try:
        if "search_mode" in sent:
            try:
                runtime_retrieval.set_search_mode(request.search_mode)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            log.info("runtime_retrieval_changed", key="search_mode", new_value=request.search_mode)
        if "rerank" in sent:
            runtime_retrieval.set_rerank(request.rerank)
            log.info("runtime_retrieval_changed", key="rerank", new_value=request.rerank)
    except RuntimeConfigUnavailable as exc:
        log.error("runtime_retrieval_write_failed", error=str(exc)[:200])
        raise HTTPException(status_code=503, detail="Runtime config store unavailable") from exc

    return _payload(runtime_retrieval, settings)
