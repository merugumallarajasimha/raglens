"""Paper management API routes."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.database.postgres import get_db_session
from app.ingestion.pipeline import ingest_paper, ingest_papers_from_directory

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
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
) -> PaperUploadResponse:
    """Upload a single PDF for ingestion."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        paper = ingest_paper(tmp_path, db=db)
    except Exception as e:
        logger.error(f"Upload ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest paper: {e}",
        ) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return PaperUploadResponse(
        paper_id=paper.metadata.paper_id,
        title=paper.metadata.title,
        page_count=paper.page_count,
        sections=len(paper.sections),
    )


@router.post("/ingest", response_model=dict)
async def ingest_directory(
    request: IngestDirectoryRequest,
    db: Session = Depends(get_db_session),
) -> dict:
    """Ingest all PDFs from a directory."""
    directory = Path(request.directory)
    if not directory.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {directory}",
        )

    try:
        papers = ingest_papers_from_directory(str(directory), db=db)
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    results = [
        {
            "paper_id": paper.metadata.paper_id,
            "title": paper.metadata.title,
            "pages": paper.page_count,
            "sections": len(paper.sections),
        }
        for paper in papers
    ]

    return {
        "ingested": len(papers),
        "results": results,
    }


@router.get("/{paper_id}", response_model=dict)
async def get_paper(
    paper_id: str,
    db: Session = Depends(get_db_session),
) -> dict:
    """Get a specific paper by ID."""
    from app.database.models import Paper

    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper not found: {paper_id}",
        )

    return {
        "id": paper.id,
        "title": paper.title,
        "abstract": paper.abstract,
        "year": paper.year,
        "source": paper.source,
        "url": paper.url,
    }


@router.get("/", response_model=dict)
async def list_papers(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db_session),
) -> dict:
    """List all ingested papers."""
    from app.database.models import Paper

    papers = db.query(Paper).offset(skip).limit(limit).all()
    results = [
        {
            "id": p.id,
            "title": p.title,
            "year": p.year,
        }
        for p in papers
    ]
    return {"papers": results, "count": len(results)}


@router.delete("/{paper_id}", response_model=dict)
async def delete_paper(
    paper_id: str,
    db: Session = Depends(get_db_session),
) -> dict:
    """Delete a paper and all associated data."""
    from app.database.models import Paper

    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper not found: {paper_id}",
        )

    db.delete(paper)
    db.commit()
    return {"deleted": paper_id}