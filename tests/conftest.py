"""Shared test fixtures for RAGLens tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from unittest.mock import patch

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory for test-generated data."""
    d = Path(__file__).parent.parent / "data" / "evaluation"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def sample_pdf_path(test_data_dir: Path):
    """Generate a synthetic test paper PDF once per session."""
    from tests.conftest_helpers import create_test_paper_pdf

    pdf_path = test_data_dir / "test_sample_paper.pdf"
    if not pdf_path.exists():
        pdf_path = Path(create_test_paper_pdf(
            pdf_path,
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors="J. Smith, A. Johnson, M. Chen",
            year=2024,
        ))
    return pdf_path


@pytest.fixture(scope="session")
def sample_pdf_path_2(test_data_dir: Path):
    """Generate a second synthetic test paper for multi-paper tests."""
    from tests.conftest_helpers import create_test_paper_pdf

    pdf_path = test_data_dir / "test_sample_paper_2.pdf"
    if not pdf_path.exists():
        pdf_path = Path(create_test_paper_pdf(
            pdf_path,
            title="Self-RAG: Self-Reflective Retrieval-Augmented Generation",
            authors="L. Peng, S. Guizhou",
            year=2023,
        ))
    return pdf_path


@pytest.fixture(scope="function")
def in_memory_db() -> Iterator[Session]:
    """Provide an in-memory SQLite session for testing.

    Each test gets a fresh database with all tables created.
    """
    from app.database.models import Base as ModelBase
    from app.database.postgres import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    ModelBase.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

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
