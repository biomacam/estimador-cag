"""``POST /v1/estimate/stages/*`` — the RAG pipeline, one stage at a time.

The full ``/v1/estimate/from-transcript`` pipeline hides its intermediate
artifacts, so this router exposes each stage as a stateless endpoint that REUSES
the same pure functions as the orchestrator: the caller persists each stage's
output and passes it to the next one.

Same posture as the sibling routers (422 / 401 / 429 / 502). The estimate key
guards the whole set. The generate stage deliberately does NOT replicate the
orchestrator's corrective retries: it returns the grounding signals (fabricated
ids, coherence) so a UI can show them.
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import get_settings
from app.dependencies import get_embedder, get_runtime_retrieval_config, get_token_encoder
from app.rag.context_assembler import build_context_block, truncate_to_token_budget
from app.rag.errors import RagError, RetrievalError
from app.rag.estimator import generate_estimate
from app.rag.observability import get_request_id, log_stage
from app.rag.query_reformulator import compose_search_text, reformulate_query
from app.rag.rate_limiting import limiter
from app.rag.retrieval.pipeline import retrieve as retrieve_chunks
from app.rag.schemas import (
    AssembleRequest,
    AssembleResult,
    GenerateRequest,
    GenerateResult,
    ReformulateRequest,
    ReformulationResult,
    RetrievalRequest,
    RetrievalResult,
)
from app.rag.security import require_estimate_key
from app.rag.validation import check_coherence, validate_citations

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate/stages", tags=["estimate-stages"])


@router.post(
    "/reformulate",
    response_model=ReformulationResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("30/minute")
async def reformulate(request: Request, payload: ReformulateRequest) -> ReformulationResult:
    """Stage 1 — distill a transcript into a structured brief + search text."""
    request_id = get_request_id(request)
    try:
        with log_stage("reformulation", request_id):
            query = await reformulate_query(payload.transcript)
            search_text = compose_search_text(query)
        return ReformulationResult(query=query, search_text=search_text)
    except RagError as exc:
        log.error("stage_failed", stage="reformulation", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Query reformulation failed.") from exc


@router.post(
    "/retrieve",
    response_model=RetrievalResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def retrieve(request: Request, payload: RetrievalRequest) -> RetrievalResult:
    """Stage 2 — embed the search text and run vector/hybrid retrieval.

    Returns the same ``RetrievalResult`` as ``/v1/retrieval/search`` so the caller
    sees ``low_confidence``/``candidates_evaluated`` for the soft-fail branch.
    Search mode and reranking resolve from request, runtime override, then settings."""
    request_id = get_request_id(request)
    embedder = get_embedder()
    if embedder is None:
        log.error("stage_failed", stage="retrieval", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    settings = get_settings()
    runtime = get_runtime_retrieval_config()
    search_mode = payload.search_mode or runtime.effective_search_mode()
    rerank = payload.rerank if payload.rerank is not None else runtime.effective_rerank()

    try:
        with log_stage("retrieval", request_id, sectors=payload.sectors):
            query_embedding = await asyncio.to_thread(embedder.embed_one, payload.query_text)
            return await retrieve_chunks(
                query_embedding=query_embedding,
                query_text=payload.query_text,
                search_mode=search_mode,
                rerank=rerank,
                top_k=payload.top_k,
                recall_k=settings.RETRIEVAL_RECALL_TOP_K,
                rerank_top_n=settings.RERANK_TOP_N,
                rrf_k=settings.RRF_K,
                distance_threshold=payload.distance_threshold,
                sectors=payload.sectors,
                project_year_min=payload.project_year_min,
                project_year_max=payload.project_year_max,
                chunk_types=payload.chunk_types,
            )
    except RetrievalError as exc:
        raise HTTPException(status_code=502, detail="Retrieval failed.") from exc
    except Exception as exc:  # noqa: BLE001 — embedding/other failures -> 502.
        log.error("stage_failed", stage="retrieval", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Failed to run retrieval.") from exc


@router.post(
    "/assemble",
    response_model=AssembleResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def assemble(request: Request, payload: AssembleRequest) -> AssembleResult:
    """Stage 3 — truncate to the token budget (whole chunks) and build the ``<source>`` block."""
    request_id = get_request_id(request)
    budget = payload.max_context_tokens or get_settings().MAX_CONTEXT_TOKENS
    encoder = get_token_encoder()

    with log_stage("augmentation", request_id, budget=budget):
        kept = truncate_to_token_budget(payload.chunks, budget, encoder)
        context_block = build_context_block(kept)
        token_count = len(encoder.encode(context_block))

    return AssembleResult(
        context_block=context_block,
        kept_chunks=kept,
        dropped_count=len(payload.chunks) - len(kept),
        token_count=token_count,
    )


@router.post(
    "/generate",
    response_model=GenerateResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def generate(request: Request, payload: GenerateRequest) -> GenerateResult:
    """Stage 4 — generate the grounded estimate and report grounding signals."""
    request_id = get_request_id(request)
    try:
        with log_stage("generation", request_id, sources=len(payload.kept_chunks)):
            estimate = await generate_estimate(payload.context_block, structured_query=payload.query)
    except RagError as exc:
        log.error("stage_failed", stage="generation", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Estimate generation failed.") from exc

    return GenerateResult(
        estimate=estimate,
        fabricated_source_ids=validate_citations(estimate, payload.kept_chunks),
        coherent=check_coherence(estimate),
    )
