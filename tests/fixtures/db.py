"""Test database fixtures using SQLite in-memory.

These fixtures provide a real SQLAlchemy engine backed by SQLite for
unit tests, avoiding the need for a running PostgreSQL instance.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base as ModelBase
from app.database.postgres import Base


@pytest.fixture(scope="function")
def in_memory_db() -> Iterator[Session]:
    """Provide an in-memory SQLite session for testing.

    Each test gets a fresh database with all tables created.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    # Create all tables
    Base.metadata.create_all(engine)
    ModelBase.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    # Patch get_db_session to use our test session
    def _get_test_session():
        session = TestingSessionLocal()
        yield session

    with patch("app.database.postgres.get_db_session", _get_test_session):
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()
            engine.dispose()
