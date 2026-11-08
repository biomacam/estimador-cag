"""``POST /v1/estimate/from-transcript`` — transcript -> grounded estimate.

The strictest endpoint on the service: a tight rate limit (10/min) and
idempotency, because each call drives a multi-step LLM pipeline.

Thin transport: validation in ``EstimateRequest`` (422), auth (401), rate limiting
(429). Any pipeline failure becomes a 502 (the request id is on ``X-Request-ID``).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.rag.errors import RagError
from app.rag.estimator import estimate_from_transcript
from app.rag.rate_limiting import limiter
from app.rag.schemas import Estimate, EstimateRequest
from app.rag.security import require_estimate_key

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate", tags=["estimate"])


@router.post(
    "/from-transcript",
    response_model=Estimate,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def from_transcript(request: Request, payload: EstimateRequest) -> Estimate:
    """Produce a grounded estimate from a raw transcript (idempotent on key)."""
    try:
        return await estimate_from_transcript(
            payload.transcript,
            idempotency_key=payload.idempotency_key,
        )
    except RagError as exc:
        log.error("estimate_failed", error_type=type(exc).__name__, error=str(exc)[:300])
        raise HTTPException(status_code=502, detail="Failed to produce an estimate.") from exc
