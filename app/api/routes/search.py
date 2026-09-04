"""Search API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.config import get_settings

router = APIRouter()
logger = get_logger("api.search")


class SearchRequest(BaseModel):
    """Request model for paper search."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = None
    paper_ids: Optional[list[str]] = None
    sources: Optional[list[str]] = None
    year_range: Optional[tuple[int, int]] = None
    sections: Optional[list[str]] = None


class SearchResult(BaseModel):
    """A single search result."""

    chunk_id: str
    paper_id: str
    title: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str
    score: float


@router.post("/search", response_model=dict)
async def search(request: SearchRequest) -> dict:
    """Search the paper corpus using hybrid retrieval.

    This endpoint will be fully implemented in Phase 8 (hybrid retrieval).
    """
    settings = get_settings()
    top_k = request.top_k or settings.hybrid_top_k

    return {
        "query": request.query,
        "results": [],
        "top_k": top_k,
        "total": 0,
        "message": "Search backend not yet implemented",
    }


@router.get("/papers/{paper_id}/chunks", response_model=dict)
async def get_chunks(
    paper_id: str,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Get chunks for a specific paper.

    This endpoint will be fully implemented in Phase 2 (chunking).
    """
    return {
        "paper_id": paper_id,
        "chunks": [],
        "limit": limit,
        "offset": offset,
        "message": "Chunk retrieval not yet implemented",
    }
