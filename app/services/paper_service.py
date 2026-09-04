"""Paper service — manages paper lifecycle and persistence.

Provides high-level operations for ingesting papers, saving metadata
to PostgreSQL, and coordinating with the vector store and embedding
provider.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import IngestionError
from app.database.models import Paper
from app.database.repositories import PaperRepository, ChunkRepository
from app.ingestion.models import StructuredPaper
from app.ingestion.chunker import Chunk, Chunker

logger = logging.getLogger("raglens.services.paper")


class PaperService:
    """Service for managing paper ingestion and retrieval.

    Args:
        settings: Application settings.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    def save_paper(self, paper: StructuredPaper) -> dict:
        """Save or update a paper in the database.

        Handles deduplication: if the paper already exists, it is not re-ingested.

        Args:
            paper: The structured paper to save.

        Returns:
            Dict with paper metadata.
        """
        repo = PaperRepository()

        # Check if paper already exists
        if repo.exists(paper.metadata.paper_id):
            logger.info(
                "Paper already exists, skipping",
                extra={"extra_data": {"paper_id": paper.metadata.paper_id}},
            )
            return repo.get_by_id_dict(paper.metadata.paper_id)

        paper_record = repo.create(
            paper_id=paper.metadata.paper_id,
            title=paper.metadata.title,
            abstract=paper.metadata.abstract,
            year=paper.metadata.year,
            source=paper.metadata.source,
            url=paper.metadata.url,
            authors=paper.metadata.authors,
        )

        logger.info(
            "Paper saved to database",
            extra={"extra_data": {
                "paper_id": paper.metadata.paper_id,
                "title": paper.metadata.title,
            }},
        )

        return repo._paper_to_dict(paper_record)

    def save_chunks(self, chunks: list[Chunk]) -> int:
        """Save chunks to the database.

        Args:
            chunks: List of Chunk objects to save.

        Returns:
            Number of chunks saved (excluding duplicates).
        """
        chunk_repo = ChunkRepository()

        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text": chunk.text,
                "token_count": chunk.token_count,
            })

        count = chunk_repo.bulk_create(chunks_data)
        logger.info(
            "Chunks saved to database",
            extra={"extra_data": {
                "count": count,
                "paper_id": chunks[0].paper_id if chunks else "unknown",
            }},
        )
        return count

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """Retrieve a paper by ID."""
        repo = PaperRepository()
        return repo.get_by_id_dict(paper_id)

    def get_chunks(self, paper_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        """Retrieve chunks for a paper."""
        chunk_repo = ChunkRepository()
        return chunk_repo.list_by_paper(paper_id, limit=limit, offset=offset)

    def chunk_paper(self, paper: StructuredPaper) -> list[Chunk]:
        """Chunk a structured paper using configured settings."""
        chunker = Chunker(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            strategy=self._settings.chunking_strategy,
        )
        return chunker.chunk_paper(paper)

    def ingest_and_store(self, file_path: str) -> dict:
        """Full pipeline: ingest PDF, chunk, save to database.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dict with ingestion summary.
        """
        from app.ingestion.pipeline import ingest_paper

        # Step 1: Ingest PDF
        paper = ingest_paper(file_path)

        # Step 2: Save paper metadata
        self.save_paper(paper)

        # Step 3: Chunk
        chunks = self.chunk_paper(paper)

        # Step 4: Save chunks
        chunk_count = self.save_chunks(chunks)

        return {
            "paper_id": paper.metadata.paper_id,
            "title": paper.metadata.title,
            "pages": paper.page_count,
            "chunks": chunk_count,
            "sections": len(paper.sections),
        }
