"""Ingest API routes — PDF upload → parse → chunk → embed → Qdrant upsert."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.deps import get_embedding_provider, get_vector_store, get_settings
from app.core.config import Settings
from app.core.logging import get_logger
from app.ingestion.chunker import Chunker
from app.ingestion.pipeline import ingest_paper
from app.retrieval.embeddings import EmbeddingProvider

router = APIRouter()
logger = get_logger("api.ingest")


class IngestFileResponse(BaseModel):
    """Response model for /ingest/file."""

    status: str
    chunks_ingested: int
    collection_points_count: int


@router.post("/file", response_model=IngestFileResponse)
async def ingest_file(
    file: UploadFile = File(...),
    vector_store=Depends(get_vector_store),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    settings: Settings = Depends(get_settings),
) -> IngestFileResponse:
    """Upload a PDF, parse + chunk it, embed it, and upsert into Qdrant.

    Uses the same vector store and embedding provider the /query/query
    endpoint reads from, so writes are visible to retrieval immediately.
    """
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
        # 1. Parse PDF into a StructuredPaper
        paper = ingest_paper(tmp_path)

        # 2. Chunk the paper using the configured chunker
        chunker = Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            strategy=settings.chunking_strategy,
        )
        chunks = chunker.chunk_paper(paper)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No chunks were produced from the PDF.",
            )

        # 3. Generate embeddings via the shared embedding provider
        embeddings = embedding_provider.embed_documents([c.text for c in chunks])

        # 4. Upsert into the same Qdrant collection /query reads from
        chunks_ingested = vector_store.upsert_chunks(chunks, embeddings)

        # 5. Confirm the collection point count
        collection_points_count = vector_store.count()

        logger.info(
            "File ingestion complete",
            extra={"extra_data": {
                "paper_id": paper.metadata.paper_id,
                "chunks_ingested": chunks_ingested,
                "collection_points_count": collection_points_count,
            }},
        )

        return IngestFileResponse(
            status="success",
            chunks_ingested=chunks_ingested,
            collection_points_count=collection_points_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest file: {e}",
        ) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)