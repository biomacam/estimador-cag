"""SQLAlchemy engine, session factory and per-request session helpers.

Synchronous SQLAlchemy 2.0: ingestion runs as a BackgroundTask or a one-shot
admin operation, never on the estimation hot path.

The engine and session factory are built on first use, not at import time, so
importing this package (and starting the app) never requires Postgres.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def create_engine_from_settings() -> Engine:
    """Build the global engine from ``Settings.DATABASE_URL`` (singleton)."""
    return create_engine(get_settings().DATABASE_URL, pool_pre_ping=True)


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=create_engine_from_settings(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def new_session() -> Session:
    """Open a standalone Session; the caller closes it (e.g. in a BackgroundTask)."""
    return _session_factory()()


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a Session and closes it on exit."""
    session = new_session()
    try:
        yield session
    finally:
        session.close()


@lru_cache
def create_async_engine_from_settings() -> AsyncEngine:
    """Build an async PostgreSQL engine from the shared database URL."""
    database_url = make_url(get_settings().DATABASE_URL)
    if database_url.get_backend_name() == "postgresql":
        database_url = database_url.set(drivername="postgresql+asyncpg")
    return create_async_engine(database_url, pool_pre_ping=True)


@lru_cache
def _async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=create_async_engine_from_settings(),
        autoflush=False,
        expire_on_commit=False,
    )


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Yield an async session for transactional embedding ingestion."""
    async with _async_session_factory()() as session:
        yield session
