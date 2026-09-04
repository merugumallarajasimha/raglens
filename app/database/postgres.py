"""PostgreSQL database connection and session management."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from app.core.config import get_settings

_engine: object = None
_SessionLocal: object = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.postgres_url,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"connect_timeout": "5"},
        )
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _engine, _SessionLocal


class Base(DeclarativeBase):
    pass


def get_db_session() -> Iterator[Session]:
    """FastAPI dependency that yields a database session."""
    if _SessionLocal is None:
        _get_engine()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they do not exist."""
    engine, _ = _get_engine()
    Base.metadata.create_all(bind=engine)
