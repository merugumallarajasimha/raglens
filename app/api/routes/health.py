"""Health check endpoint."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Response model for the basic health endpoint."""

    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = Field(..., description="Application version")


class DetailedHealthResponse(BaseModel):
    """Response model for the detailed health endpoint."""

    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = Field(..., description="Application version")
    checks: dict[str, Any] = Field(default_factory=dict)


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check — returns healthy if the application is running."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.project_version,
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check() -> DetailedHealthResponse:
    """Detailed health check — verifies connectivity to all dependencies."""
    settings = get_settings()

    checks: dict[str, Any] = {}

    # Check Qdrant connectivity
    try:
        from app.database.qdrant import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
        checks["qdrant"] = "healthy"
    except Exception as e:
        checks["qdrant"] = f"unavailable: {type(e).__name__}"

    # Check PostgreSQL connectivity
    try:
        from app.database.postgres import get_db_session
        session = next(get_db_session())
        session.execute("SELECT 1")
        session.close()
        checks["postgres"] = "healthy"
    except Exception as e:
        checks["postgres"] = f"unavailable: {type(e).__name__}"

    # Check Ollama connectivity
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{settings.ollama_url}/api/tags")
            if resp.status_code == 200:
                checks["ollama"] = "healthy"
            else:
                checks["ollama"] = "unavailable"
    except Exception:
        checks["ollama"] = "unavailable"

    all_healthy = all(v == "healthy" for v in checks.values())
    overall = "healthy" if all_healthy else "degraded"

    return DetailedHealthResponse(
        status=overall,
        version=settings.project_version,
        checks=checks,
    )


@router.get("/liveness")
async def liveness() -> dict[str, str]:
    """Kubernetes-style liveness probe — always returns OK if the app is running."""
    return {"status": "alive"}
