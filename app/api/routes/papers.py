"""Paper management API routes."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.ingestion.pipeline import ingest_paper

router = APIRouter()
logger = get_logger("api.papers")


class PaperUploadResponse(BaseModel):
    """Response model for paper upload."""

    paper_id: str
    title: str
    page_count: int
    sections: int


class IngestDirectoryRequest(BaseModel):
    """Request to ingest a directory of PDFs."""

    directory: str = Field(..., description="Directory path containing PDFs")


@router.post("/upload", response_model=PaperUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_paper(file: UploadFile = File(...)) -> PaperUploadResponse:
    """Upload a single PDF for ingestion.

    Args:
        file: PDF file to upload.

    Returns:
        Paper metadata summary.

    Raises:
        400: If the file is not a PDF.
        500: If ingestion fails.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        paper = ingest_paper(tmp_path)
    except Exception as e:
        logger.error(f"Upload ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest paper: {e}",
        ) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info(
        "Paper uploaded and ingested",
        extra={"extra_data": {
            "paper_id": paper.metadata.paper_id,
            "title": paper.metadata.title,
            "pages": paper.page_count,
        }},
    )

    return PaperUploadResponse(
        paper_id=paper.metadata.paper_id,
        title=paper.metadata.title,
        page_count=paper.page_count,
        sections=len(paper.sections),
    )


@router.post("/ingest", response_model=dict)
async def ingest_directory(request: IngestDirectoryRequest) -> dict:
    """Ingest all PDFs from a directory.

    Args:
        request: Request containing the directory path.

    Returns:
        Summary of ingestion results.
    """
    from app.ingestion.pipeline import ingest_papers_from_directory

    directory = Path(request.directory)
    if not directory.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {directory}",
        )

    try:
        papers = ingest_papers_from_directory(str(directory))
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    results = []
    for paper in papers:
        results.append({
            "paper_id": paper.metadata.paper_id,
            "title": paper.metadata.title,
            "pages": paper.page_count,
            "sections": len(paper.sections),
        })

    return {
        "ingested": len(papers),
        "results": results,
    }


@router.get("/{paper_id}", response_model=dict)
async def get_paper(paper_id: str) -> dict:
    """Get a specific paper by ID."""
    from app.database.repositories import PaperRepository

    repo = PaperRepository()
    paper = repo.get_by_id(paper_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper not found: {paper_id}",
        )
    return paper


@router.get("/", response_model=dict)
async def list_papers(skip: int = 0, limit: int = 100) -> dict:
    """List all ingested papers."""
    from app.database.repositories import PaperRepository

    repo = PaperRepository()
    try:
        papers = repo.list(skip=skip, limit=limit)
    except Exception as e:
        return {
            "papers": [],
            "count": 0,
            "error": f"Database unavailable: {type(e).__name__}",
        }
    return {"papers": papers, "count": len(papers)}


@router.delete("/{paper_id}", response_model=dict)
async def delete_paper(paper_id: str) -> dict:
    """Delete a paper and all its chunks."""
    from app.database.repositories import PaperRepository

    repo = PaperRepository()
    if not repo.delete(paper_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper not found: {paper_id}",
        )
    return {"deleted": paper_id}
