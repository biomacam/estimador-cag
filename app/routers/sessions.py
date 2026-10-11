from collections.abc import Callable
from typing import TypeVar

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.attachments import (
    AttachmentError,
    AttachmentExtractionError,
    AttachmentTooLargeError,
    UnsupportedAttachmentError,
    enrich_transcript,
    extract_text,
    safe_filename,
)
from app.config import Settings, get_settings
from app.dependencies import get_estimation_service, get_session_store
from app.guardrails.input import InputGuardrailViolation
from app.prompts.loader import PromptVersionNotFoundError
from app.schemas.estimation import (
    ACBResponse,
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.schemas.session import CreateSessionResponse, SessionInfoResponse
from app.services.llm_errors import UNEXPECTED_ERROR
from app.services.llm_service import LLM_PROVIDER_ERROR_MESSAGE, EstimationService, LLMServiceError
from app.services.llm_wrapper import LLMConfigurationError
from app.sessions import Session, SessionNotFoundError, SessionStore
from app.sessions.tier_resolver import Tier

log = structlog.get_logger()

_T = TypeVar("_T")

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse, status_code=201)
def create_session(store: SessionStore = Depends(get_session_store)) -> CreateSessionResponse:
    """Create an empty conversational session and return its identifier."""
    session = store.create()
    log.info("session_created", session_id=session.session_id)
    return CreateSessionResponse(session_id=session.session_id)


@router.get(
    "/{session_id}",
    response_model=SessionInfoResponse,
    responses={404: {"description": "Unknown session_id"}},
)
def get_session(
    session_id: str, store: SessionStore = Depends(get_session_store)
) -> SessionInfoResponse:
    """Current project metadata and history size of a session (handy for debugging)."""
    try:
        session = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return SessionInfoResponse(
        session_id=session.session_id,
        message_count=len(session.history.messages),
        max_turns=session.history.max_turns,
        anchors_count=len(session.history.anchors),
        summary_chars=len(session.history.summary or ""),
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
        metadata=session.metadata,
    )


_TURN_RESPONSES: dict[int | str, dict[str, str]] = {
    400: {"description": "Rejected by an input guardrail (moderation, prompt injection or PII)"},
    404: {"description": "Unknown session_id"},
    413: {"description": "An attachment exceeds the size limit"},
    415: {"description": "An attachment is not a PDF or DOCX"},
    422: {"description": "Invalid request, unreadable attachment or unknown prompt version"},
    502: {"description": "LLM generation failed; the body adds reason.code (and a safe reason.message)"},
    503: {"description": "No LLM provider credentials configured"},
}


@router.post("/{session_id}/estimate", response_model=EstimationResponse, responses=_TURN_RESPONSES)
def estimate_in_session(
    session_id: str,
    transcript: str = Form(min_length=20, max_length=50_000),
    attachments: list[UploadFile] = File(default_factory=list),
    project_type: ProjectType = Form(ProjectType.WEB_SAAS),
    detail_level: DetailLevel = Form(DetailLevel.MEDIUM),
    output_format: OutputFormat = Form(OutputFormat.PHASES_TABLE),
    tier: Tier | None = Form(None, description="Audience tier override; derived from the conversation when omitted"),
    prompt_version: str = Query(default="v1", description="Version of the estimation prompt templates"),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
    settings: Settings = Depends(get_settings),
) -> EstimationResponse:
    """Extract the text of each PDF/DOCX attachment locally, append it to the transcript
    under a ``--- attachment: <name> ---`` separator and run one turn of the session
    (history window and project metadata are updated).

    Plain ``def`` for the same reason as ``create_estimation``: the LLM call blocks.
    """
    session, enriched = _prepare_turn(session_id, transcript, attachments, store, settings)
    return _run_turn(
        lambda: service.estimate_conversational(
            session=session,
            transcript=enriched,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=tier,
            version=prompt_version,
        )
    )


@router.post("/{session_id}/estimate-acb", response_model=ACBResponse, responses=_TURN_RESPONSES)
def estimate_in_session_acb(
    session_id: str,
    transcript: str = Form(min_length=20, max_length=50_000),
    attachments: list[UploadFile] = File(default_factory=list),
    project_type: ProjectType = Form(ProjectType.WEB_SAAS),
    detail_level: DetailLevel = Form(DetailLevel.MEDIUM),
    output_format: OutputFormat = Form(OutputFormat.PHASES_TABLE),
    tier: Tier | None = Form(None, description="Audience tier override; derived from the conversation when omitted"),
    prompt_version: str = Query(default="v1", description="Version of the estimation prompt templates"),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
    settings: Settings = Depends(get_settings),
) -> ACBResponse:
    """Actor-Critic-Boss variant of ``/estimate``: same multipart contract.

    A Critic audits each draft and a Boss decides whether to accept it, retry the actor
    with the critic's feedback, or return the last draft annotated with open caveats.
    The ``acb`` field carries the per-iteration audit trail.
    """
    session, enriched = _prepare_turn(session_id, transcript, attachments, store, settings)
    return _run_turn(
        lambda: service.estimate_with_acb(
            session=session,
            transcript=enriched,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=tier,
            version=prompt_version,
        )
    )


def _prepare_turn(
    session_id: str,
    transcript: str,
    attachments: list[UploadFile],
    store: SessionStore,
    settings: Settings,
) -> tuple[Session, str]:
    """Resolve the session and return it with the attachment-enriched transcript."""
    try:
        session = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc

    uploads = [upload for upload in attachments if upload.filename]
    if len(uploads) > settings.MAX_ATTACHMENTS:
        raise HTTPException(
            status_code=422, detail=f"At most {settings.MAX_ATTACHMENTS} attachments are allowed."
        )

    try:
        extracted = [
            (safe_filename(upload.filename), _extract_upload(upload, settings)) for upload in uploads
        ]
    except UnsupportedAttachmentError as exc:
        raise _attachment_http_error(415, exc) from exc
    except AttachmentTooLargeError as exc:
        raise _attachment_http_error(413, exc) from exc
    except AttachmentExtractionError as exc:
        raise _attachment_http_error(422, exc) from exc

    enriched = enrich_transcript(transcript, extracted)
    log.info(
        "session_estimate_received",
        session_id=session_id,
        transcript_chars=len(transcript),
        enriched_chars=len(enriched),
        attachments=[name for name, _ in extracted],
    )
    return session, enriched


def _run_turn(run: Callable[[], _T]) -> _T:
    """Map pipeline exceptions to HTTP errors, identically for both turn endpoints."""
    try:
        return run()
    except InputGuardrailViolation as exc:
        log.info("estimation_blocked_by_input_guardrail", reason=exc.reason, message=exc.message)
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        ) from exc
    except PromptVersionNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LLMConfigurationError, LLMServiceError):
        raise
    except Exception as exc:
        log.error("session_estimate_error", error_type=type(exc).__name__)
        raise LLMServiceError(LLM_PROVIDER_ERROR_MESSAGE, reason=UNEXPECTED_ERROR) from exc


def _extract_upload(upload: UploadFile, settings: Settings) -> str:
    name = safe_filename(upload.filename or "")
    # Read one byte past the limit to detect oversize without loading the whole file.
    content = upload.file.read(settings.MAX_ATTACHMENT_BYTES + 1)
    if len(content) > settings.MAX_ATTACHMENT_BYTES:
        raise AttachmentTooLargeError(
            name, f"Attachment exceeds {settings.MAX_ATTACHMENT_BYTES} bytes."
        )
    return extract_text(name, content, max_chars=settings.MAX_ATTACHMENT_CHARS)


def _attachment_http_error(status_code: int, exc: AttachmentError) -> HTTPException:
    return HTTPException(
        status_code=status_code, detail={"filename": exc.filename, "message": exc.message}
    )
