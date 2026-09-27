import structlog
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.routers import estimations
from app.services.llm_wrapper import LLMConfigurationError


def configure_logging() -> None:
    """Set up structlog: JSON in production, human-readable in development."""
    settings = get_settings()

    if settings.APP_ENV == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()
    log = structlog.get_logger()
    settings = get_settings()
    log.info("application_started", environment=settings.APP_ENV)
    yield
    log.info("application_shutdown")


app = FastAPI(
    title="Software Estimation CAG Service",
    description="AI-powered software estimation service using Cache Augmented Generation architecture",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(estimations.router)


@app.exception_handler(LLMConfigurationError)
async def llm_configuration_error_handler(
    request: Request, exc: LLMConfigurationError
) -> JSONResponse:
    """Translate a missing API key into a 503, whether raised in a route body or
    while resolving the ``get_llm_wrapper`` dependency (streaming endpoint)."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# Serves the static SSE demo page from app/static/, if present.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)) -> dict:
    """Return service health status. Always 200 — never gated on LLM configuration,
    since this is the endpoint operators use to tell "missing secret" from "broken code"."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": settings.APP_ENV,
        "llm_configured": settings.is_llm_configured,
    }
