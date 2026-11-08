"""API-key authentication for the RAG routers.

Two independent keys protect the two routers: a ``RETRIEVAL_API_KEY`` holder cannot
call the estimate endpoints and vice versa. Keys are compared with
``secrets.compare_digest`` (constant-time), never ``==``.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.config import get_settings

_API_KEY_HEADER = "X-API-Key"


def _verify(provided: str | None, expected: str | None) -> None:
    """Raise 401 unless ``provided`` matches the configured ``expected`` key."""
    if not expected or not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": _API_KEY_HEADER},
        )


async def require_retrieval_key(
    x_api_key: str | None = Header(default=None, alias=_API_KEY_HEADER),
) -> None:
    """Guards ``POST /v1/retrieval/search``."""
    _verify(x_api_key, get_settings().RETRIEVAL_API_KEY)


async def require_estimate_key(
    x_api_key: str | None = Header(default=None, alias=_API_KEY_HEADER),
) -> None:
    """Guards ``POST /v1/estimate/*``."""
    _verify(x_api_key, get_settings().ESTIMATE_API_KEY)
