
#librería estándar de Python para programación asíncrona mediante async y await. Gestiona un bucle de eventos, que permite coordinar operaciones concurrentes sin bloquear continuamente el hilo principal.
import asyncio
#AsyncIterator se utiliza para indicar mediante tipado que una función produce valores de manera progresiva y asíncrona.
from collections.abc import AsyncIterator

import structlog
#Depends implementa el sistema de inyección de dependencias de FastAPI.
from fastapi import APIRouter, Depends, HTTPException
from litellm.exceptions import APIError as LLMProviderAPIError
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_llm_wrapper
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    StreamEstimationRequest,
)
from app.services.evaluation import (
    evaluate_estimation_structure,
    extract_declared_total_cost,
    inject_annual_maintenance_line,
)
from app.services.llm_service import (
    LLM_PROVIDER_ERROR_MESSAGE,
    GenerationOptions,
    LLMServiceError,
    build_system_prompt,
    frame_untrusted_input,
    generate_estimation,
    new_untrusted_data_tag,
)
from app.services.llm_wrapper import LLMTruncatedResponseError, LLMWrapper

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(request: EstimationRequest) -> EstimationResponse:
    """Receive a meeting transcription and return a software project estimation.

    Plain ``def``, not ``async def``: ``generate_estimation`` blocks on the LLM
    call, and FastAPI only runs blocking work off the event loop (in a
    threadpool) for sync handlers — an ``async def`` here would freeze every
    other request on this worker, including ``/health``, for the call's duration.
    """
    opts = GenerationOptions(
        preprocessing=request.preprocessing,
        example_format=request.example_format,
        num_examples=request.num_examples,
        use_examples=request.use_examples,
        model=request.model,
        max_tokens=request.max_tokens,
        thinking_budget=request.thinking_budget,
    )

    try:
        result = generate_estimation(request.transcription, opts)
    except LLMServiceError as exc:
        log.error("estimation_endpoint_error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    validation = (
        evaluate_estimation_structure(result["estimation"], result["finish_reason"])
        if request.evaluate
        else None
    )

    declared_total_cost = extract_declared_total_cost(result["estimation"])
    annual_maintenance = (
        round(declared_total_cost * 0.12, 2) if declared_total_cost is not None else None
    )
    if annual_maintenance is not None:
        result["estimation"] = inject_annual_maintenance_line(
            result["estimation"], annual_maintenance
        )

    return EstimationResponse(
        **result, validation=validation, annual_maintenance=annual_maintenance
    )


@router.post("/estimate/stream")
async def create_estimation_stream(
    request: StreamEstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EventSourceResponse:
    """Stream a software estimation token by token via Server-Sent Events.

    The streaming path is intentionally simpler than POST /estimate: it skips
    two-phase preprocessing and structural validation, since both fight the UX
    benefit of streaming (intermediate phase 1 tokens would leak; validation
    only makes sense over the complete text).
    """
    untrusted_data_tag = new_untrusted_data_tag()
    system_prompt = build_system_prompt(untrusted_data_tag=untrusted_data_tag)

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        chunks = wrapper.complete_stream(
            system_prompt=system_prompt,
            user_message=frame_untrusted_input(request.transcription, untrusted_data_tag),
            model_override=request.model,
            max_tokens=request.max_tokens,
        )

        def _next_chunk() -> str | None:
            try:
                return next(chunks)
            except StopIteration:
                return None
            except Exception as exc:  # noqa: BLE001 — surface as SSE error event
                log.error("estimate_stream_failed", error=str(exc), error_type=type(exc).__name__)
                raise

        full_text_parts: list[str] = []
        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    full_text_parts.append(chunk)
                    yield {"event": "token", "data": chunk}

            # Full text is only known once streaming ends, so this is appended rather than inlined.
            declared_total_cost = extract_declared_total_cost("".join(full_text_parts))
            if declared_total_cost is not None:
                annual_maintenance = round(declared_total_cost * 0.12, 2)
                yield {
                    "event": "token",
                    "data": f"\n- **Annual maintenance:** {annual_maintenance:,.2f} EUR",
                }

            yield {"event": "done", "data": "[DONE]"}
        except LLMTruncatedResponseError as exc:
            log.error("estimate_stream_truncated", error=str(exc))
            yield {"event": "error", "data": "The model stopped before completing the estimation."}
        except LLMProviderAPIError as exc:
            log.error(
                "estimate_stream_provider_error", error=str(exc), error_type=type(exc).__name__
            )
            yield {"event": "error", "data": LLM_PROVIDER_ERROR_MESSAGE}
        except Exception as exc:  # noqa: BLE001 — last-resort guard, never leak raw exception text
            log.error(
                "estimate_stream_unexpected_error", error=str(exc), error_type=type(exc).__name__
            )
            yield {
                "event": "error",
                "data": "An unexpected error occurred while generating the estimation.",
            }

    return EventSourceResponse(event_generator())
