"""FastAPI application factory and root router."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure parent directory is in python search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging


def get_router(module_name: str):
    """Safely import a router whether it is in 'app.api.routers' or 'app.api.routes'."""
    try:
        mod = __import__(f"app.api.routers.{module_name}", fromlist=["router"])
        return mod.router
    except ImportError:
        mod = __import__(f"app.api.routes.{module_name}", fromlist=["router"])
        return mod.router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    setup_logging(settings)

    app = FastAPI(
        title=settings.project_name,
        version=settings.project_version,
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings

    # Dynamically load routers to avoid path mismatches
    health_router = get_router("health")
    papers_router = get_router("papers")
    query_router = get_router("query")
    search_router = get_router("search")

    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(papers_router, prefix="/papers", tags=["papers"])
    app.include_router(query_router, prefix="/query", tags=["query"])
    app.include_router(search_router, prefix="/search", tags=["search"])

    @app.get("/", tags=["root"])
    async def root() -> dict:
        return {
            "name": settings.project_name,
            "version": settings.project_version,
            "status": "running",
            "docs": "/docs",
        }

    return app


app = create_app()