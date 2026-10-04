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
from app.schemas.estimation import DetailLevel, EstimationResponse, OutputFormat, ProjectType
from app.schemas.session import CreateSessionResponse
from app.services.llm_service import LLM_PROVIDER_ERROR_MESSAGE, EstimationService
from app.services.llm_wrapper import LLMConfigurationError
from app.sessions import SessionNotFoundError, SessionStore

log = structlog.get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse, status_code=201)
def create_session(store: SessionStore = Depends(get_session_store)) -> CreateSessionResponse:
    """Create an empty conversational session and return its identifier."""
    session = store.create()
    log.info("session_created", session_id=session.session_id)
    return CreateSessionResponse(session_id=session.session_id)


@router.post(
    "/{session_id}/estimate",
    response_model=EstimationResponse,
    responses={
        400: {"description": "Rejected by an input guardrail (moderation, prompt injection or PII)"},
        404: {"description": "Unknown session_id"},
        413: {"description": "An attachment exceeds the size limit"},
        415: {"description": "An attachment is not a PDF or DOCX"},
        422: {"description": "Invalid request, unreadable attachment or unknown prompt version"},
        502: {"description": "LLM generation or structured validation failed"},
        503: {"description": "No LLM provider credentials configured"},
    },
)
def estimate_in_session(
    session_id: str,
    transcript: str = Form(min_length=20, max_length=50_000),
    attachments: list[UploadFile] = File(default_factory=list),
    project_type: ProjectType = Form(ProjectType.WEB_SAAS),
    detail_level: DetailLevel = Form(DetailLevel.MEDIUM),
    output_format: OutputFormat = Form(OutputFormat.PHASES_TABLE),
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

    try:
        return service.estimate_conversational(
            session=session,
            transcript=enriched,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            version=prompt_version,
        )
    except InputGuardrailViolation as exc:
        log.info("estimation_blocked_by_input_guardrail", reason=exc.reason, message=exc.message)
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        ) from exc
    except PromptVersionNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMConfigurationError:
        raise
    except Exception as exc:
        log.error("session_estimate_error", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail=LLM_PROVIDER_ERROR_MESSAGE) from exc


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
