"""Repository layer for database access patterns.

Provides typed repository classes for papers, chunks, authors, and sections.
Uses SQLAlchemy ORM with the models defined in app.database.models.
"""

from __future__ import annotations

from typing import Any, Optional
from sqlalchemy import select, delete as sql_delete, insert
from sqlalchemy.orm import Session

from app.database.models import Paper, Author, Chunk, Section, paper_authors
from app.database.postgres import Base, get_db_session


class PaperRepository:
    """Repository for paper metadata operations."""

    def __init__(self, db: Optional[Session] = None) -> None:
        self._db = db

    def _get_db(self) -> Session:
        if self._db is None:
            self._db = next(get_db_session())
        return self._db

    def create(
        self,
        paper_id: str,
        title: str,
        abstract: Optional[str] = None,
        year: Optional[int] = None,
        source: str = "local",
        url: Optional[str] = None,
        authors: Optional[list[str]] = None,
    ) -> Paper:
        """Create a new paper record. Raises if paper_id already exists."""
        db = self._get_db()

        # Check for duplicates
        existing = db.get(Paper, paper_id)
        if existing:
            return existing

        paper = Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            year=year,
            source=source,
            url=url,
        )
        db.add(paper)
        db.flush()  # assign ID

        # Add authors
        if authors:
            for author_name in authors:
                author = self._get_or_create_author(author_name, db)
                paper.authors.append(author)

        db.commit()
        db.refresh(paper)
        return paper

    def get_by_id(self, paper_id: str) -> Optional[Paper]:
        """Retrieve a paper by its ID."""
        db = self._get_db()
        return db.get(Paper, paper_id)

    def get_by_id_dict(self, paper_id: str) -> Optional[dict]:
        """Retrieve a paper as a dict."""
        paper = self.get_by_id(paper_id)
        if not paper:
            return None
        return self._paper_to_dict(paper)

    def list(self, skip: int = 0, limit: int = 100) -> list[dict]:
        """List papers with pagination."""
        db = self._get_db()
        stmt = select(Paper).offset(skip).limit(limit).order_by(Paper.created_at.desc())
        result = db.execute(stmt)
        return [self._paper_to_dict(p) for p in result.scalars().all()]

    def search_by_title(self, query: str, limit: int = 10) -> list[dict]:
        """Search papers by title (case-insensitive)."""
        db = self._get_db()
        stmt = (
            select(Paper)
            .where(Paper.title.ilike(f"%{query}%"))
            .limit(limit)
        )
        result = db.execute(stmt)
        return [self._paper_to_dict(p) for p in result.scalars().all()]

    def delete(self, paper_id: str) -> bool:
        """Delete a paper by ID. Returns True if deleted."""
        db = self._get_db()
        paper = db.get(Paper, paper_id)
        if not paper:
            return False
        db.delete(paper)
        db.commit()
        return True

    def exists(self, paper_id: str) -> bool:
        """Check if a paper exists."""
        db = self._get_db()
        return db.get(Paper, paper_id) is not None

    def _get_or_create_author(self, name: str, db: Session) -> Author:
        """Get an existing author by name or create a new one."""
        stmt = select(Author).where(Author.name == name)
        result = db.execute(stmt)
        author = result.scalars().first()
        if not author:
            author = Author(name=name)
            db.add(author)
            db.flush()
        return author

    def _paper_to_dict(self, paper: Paper) -> dict:
        """Convert a Paper ORM model to a dict."""
        return {
            "paper_id": paper.id,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "source": paper.source,
            "url": paper.url,
            "authors": [a.name for a in paper.authors],
            "created_at": paper.created_at.isoformat() if paper.created_at else None,
        }


class ChunkRepository:
    """Repository for chunk operations."""

    def __init__(self, db: Optional[Session] = None) -> None:
        self._db = db

    def _get_db(self) -> Session:
        if self._db is None:
            self._db = next(get_db_session())
        return self._db

    def create(
        self,
        chunk_id: str,
        paper_id: str,
        section: Optional[str] = None,
        subsection: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        text: str = "",
        token_count: int = 0,
    ) -> Chunk:
        """Create a new chunk record. Raises if chunk_id already exists."""
        db = self._get_db()

        existing = db.get(Chunk, chunk_id)
        if existing:
            return existing

        chunk = Chunk(
            id=chunk_id,
            paper_id=paper_id,
            section=section,
            subsection=subsection,
            page_start=page_start,
            page_end=page_end,
            text=text,
            token_count=token_count,
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)
        return chunk

    def bulk_create(self, chunks_data: list[dict]) -> int:
        """Create multiple chunks in a single transaction."""
        db = self._get_db()
        count = 0
        for data in chunks_data:
            existing = db.get(Chunk, data["chunk_id"])
            if existing:
                continue
            chunk = Chunk(
                id=data["chunk_id"],
                paper_id=data["paper_id"],
                section=data.get("section"),
                subsection=data.get("subsection"),
                page_start=data.get("page_start"),
                page_end=data.get("page_end"),
                text=data["text"],
                token_count=data.get("token_count", 0),
            )
            db.add(chunk)
            count += 1
        db.commit()
        return count

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by its ID."""
        db = self._get_db()
        return db.get(Chunk, chunk_id)

    def get_by_id_dict(self, chunk_id: str) -> Optional[dict]:
        """Retrieve a chunk as a dict."""
        chunk = self.get_by_id(chunk_id)
        if not chunk:
            return None
        return self._chunk_to_dict(chunk)

    def list_by_paper(self, paper_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        """List chunks for a specific paper."""
        db = self._get_db()
        stmt = (
            select(Chunk)
            .where(Chunk.paper_id == paper_id)
            .offset(offset)
            .limit(limit)
        )
        result = db.execute(stmt)
        return [self._chunk_to_dict(c) for c in result.scalars().all()]

    def delete_by_paper(self, paper_id: str) -> int:
        """Delete all chunks for a paper. Returns count deleted."""
        db = self._get_db()
        stmt = sql_delete(Chunk).where(Chunk.paper_id == paper_id)
        result = db.execute(stmt)
        db.commit()
        return result.rowcount

    def search_by_text(self, query: str, paper_id: Optional[str] = None, limit: int = 10) -> list[dict]:
        """Search chunks by text content (BM25-style, using LIKE)."""
        db = self._get_db()
        stmt = select(Chunk).where(Chunk.text.ilike(f"%{query}%"))
        if paper_id:
            stmt = stmt.where(Chunk.paper_id == paper_id)
        stmt = stmt.limit(limit)
        result = db.execute(stmt)
        return [self._chunk_to_dict(c) for c in result.scalars().all()]

    def _chunk_to_dict(self, chunk: Chunk) -> dict:
        """Convert a Chunk ORM model to a dict."""
        return {
            "chunk_id": chunk.id,
            "paper_id": chunk.paper_id,
            "section": chunk.section,
            "subsection": chunk.subsection,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "text": chunk.text,
            "token_count": chunk.token_count,
        }
