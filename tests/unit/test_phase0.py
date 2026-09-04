"""Phase 0 unit tests: configuration, logging, FastAPI health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging, get_logger


class TestConfiguration:
    """Test the configuration system."""

    def test_settings_load_defaults(self) -> None:
        settings = Settings()
        assert settings.project_name == "RAGLens"
        assert settings.project_version == "0.1.0"
        assert settings.postgres_port == 5432
        assert settings.qdrant_port == 6333
        assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.chunk_size == 800
        assert settings.chunk_overlap == 120

    def test_settings_postgres_url(self) -> None:
        settings = Settings()
        url = settings.postgres_url
        assert url.startswith("postgresql+psycopg2://")
        assert settings.postgres_user in url
        assert settings.postgres_db in url

    def test_get_settings_singleton(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "1024")
        monkeypatch.setenv("LLM_MODEL", "llama3.1")
        settings = Settings()
        assert settings.chunk_size == 1024
        assert settings.llm_model == "llama3.1"

    def test_settings_custom_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_COLLECTION", "custom_collection")
        settings = Settings()
        assert settings.qdrant_collection == "custom_collection"


class TestLogging:
    """Test the logging system."""

    def test_setup_logging_returns_logger(self) -> None:
        logger = setup_logging(get_settings())
        assert logger.name == "raglens"

    def test_get_logger_creates_child(self) -> None:
        logger = get_logger("test_module")
        assert logger.name == "raglens.test_module"

    def test_logger_is_configured(self) -> None:
        logger = get_logger("test_logging")
        assert logger.level is not None


class TestFastAPI:
    """Test the FastAPI application."""

    def test_health_endpoint_returns_ok(self) -> None:
        from app.api.main import create_app

        app = create_app()
        client = TestClient(app)
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_liveness_probe(self) -> None:
        from app.api.main import create_app

        app = create_app()
        client = TestClient(app)
        response = client.get("/health/liveness")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_detailed_health_endpoint(self) -> None:
        from app.api.main import create_app

        app = create_app()
        client = TestClient(app)
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "checks" in data
        assert data["checks"].get("qdrant") is not None
        assert data["checks"].get("postgres") is not None

    def test_root_endpoint(self) -> None:
        from app.api.main import create_app

        app = create_app()
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RAGLens"

    def test_docs_endpoint(self) -> None:
        from app.api.main import create_app

        app = create_app()
        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200


class TestStubs:
    """Test that stub endpoints return expected responses."""

    def test_papers_list_stub(self) -> None:
        from app.api.main import create_app

        app = create_app()
        client = TestClient(app)
        response = client.get("/papers/")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data

    def test_query_endpoint_stub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the /query endpoint responds (stub check with mock backends)."""
        from app.api.main import create_app
        from app.core.config import get_settings
        import app.api.deps as deps

        monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        get_settings.cache_clear()
        deps._embedding_provider = None
        deps._llm_provider = None
        deps._search_service = None
        deps._query_pipeline = None

        app = create_app()
        client = TestClient(app)
        response = client.post("/query", json={"query": "What is RAG?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
