"""FastAPI application factory and root router."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure parent directory is in python search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging


def _sanitize_math_syntax(text: str) -> str:
    """Sanitize raw LaTeX bracket delimiters into standard Markdown math syntax.

    - Converts display math \\[ ... \\] to $$ ... $$
    - Converts inline math \\( ... \\) to $ ... $
    """
    if not text:
        return text

    # Convert display brackets \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Convert inline brackets \( ... \) to $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    return text


def _sanitize_data(data: any) -> any:
    """Recursively sanitize strings containing LaTeX delimiters in dicts and lists."""
    if isinstance(data, str):
        return _sanitize_math_syntax(data)
    elif isinstance(data, dict):
        return {key: _sanitize_data(val) for key, val in data.items()}
    elif isinstance(data, list):
        return [_sanitize_data(item) for item in data]
    return data


class MathSanitizationMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically clean up raw LaTeX brackets in API JSON responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Process application/json responses
        if response.headers.get("content-type") == "application/json":
            body_bytes = b""
            async for chunk in response.body_iterator:
                body_bytes += chunk

            try:
                data = json.loads(body_bytes.decode("utf-8"))
                sanitized_data = _sanitize_data(data)
                cleaned_body = json.dumps(sanitized_data).encode("utf-8")

                # Copy headers and recalculate Content-Length
                headers = dict(response.headers)
                headers["content-length"] = str(len(cleaned_body))

                return Response(
                    content=cleaned_body,
                    status_code=response.status_code,
                    headers=headers,
                    media_type="application/json",
                )
            except Exception:
                # If parsing fails, fall back to returning original body bytes
                return Response(
                    content=body_bytes,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type="application/json",
                )

        return response


def get_router(module_name: str):
    """Safely import a router whether it is in 'app.api.routers' or 'app.api.routes'."""
    try:
        mod = __import__(f"app.api.routers.{module_name}", fromlist=["router"])
        return mod.router
    except (ImportError, AttributeError):
        try:
            mod = __import__(f"app.api.routes.{module_name}", fromlist=["router"])
            return mod.router
        except ImportError as err:
            raise ImportError(f"Could not find router for module '{module_name}' in routers or routes") from err


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

    # Middleware to automatically sanitize LaTeX formatting in JSON outputs
    app.add_middleware(MathSanitizationMiddleware)

    app.state.settings = settings

    # Dynamically load routers to avoid path mismatches
    health_router = get_router("health")
    papers_router = get_router("papers")
    query_router = get_router("query")
    search_router = get_router("search")
    ingest_router = get_router("ingest")

    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(papers_router, prefix="/papers", tags=["papers"])
    app.include_router(query_router, prefix="/query", tags=["query"])
    app.include_router(search_router, prefix="/search", tags=["search"])
    app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])

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