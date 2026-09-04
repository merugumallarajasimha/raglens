"""FastAPI application factory and root router."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging


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

    from app.api.routes.health import router as health_router
    app.include_router(health_router, prefix="/health", tags=["health"])

    from app.api.routes.papers import router as papers_router
    app.include_router(papers_router, prefix="/papers", tags=["papers"])

    from app.api.routes.query import router as query_router
    app.include_router(query_router, tags=["query"])

    from app.api.routes.search import router as search_router
    app.include_router(search_router, tags=["search"])

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
