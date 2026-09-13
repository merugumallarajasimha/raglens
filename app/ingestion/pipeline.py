"""Ingestion pipeline — orchestrates PDF parsing, structure parsing, metadata extraction, and DB persistence."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.exceptions import IngestionError, PDFParseError
from app.database.models import Author, Chunk, Paper, Section
from app.ingestion.metadata_extractor import extract_metadata
from app.ingestion.models import StructuredPaper
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.structure_parser import parse_structure

logger = logging.getLogger("raglens.ingestion")


def ingest_paper(file_path: str | Path, db: Optional[Session] = None) -> StructuredPaper:
    """Ingest a single PDF and return a fully structured paper."""
    path = Path(file_path)

    # Step 1: Parse PDF
    try:
        paper = parse_pdf(path)
    except PDFParseError:
        raise
    except FileNotFoundError:
        raise
    except Exception as e:
        raise IngestionError(
            f"Failed to parse PDF {path}", {"file": str(path), "error": str(e)}
        ) from e

    # Step 2: Extract metadata
    paper = extract_metadata(paper)

    # Step 3: Parse structure
    paper = parse_structure(paper)

    # Step 4: Save to Database if DB session is provided
    if db is not None:
        save_paper_to_db(paper, db)

    logger.info(
        "Paper ingestion complete",
        extra={"extra_data": {
            "paper_id": paper.metadata.paper_id,
            "title": paper.metadata.title,
            "pages": paper.page_count,
        }},
    )

    return paper


def ingest_papers_from_directory(
    dir_path: str | Path,
    recursive: bool = True,
    db: Optional[Session] = None,
) -> list[StructuredPaper]:
    """Ingest all PDFs from a directory and persist to DB if session provided."""
    directory = Path(dir_path)
    if not directory.exists():
        raise IngestionError(f"Directory not found: {directory}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(directory.glob(pattern))

    logger.info(
        "Starting batch ingestion",
        extra={"extra_data": {
            "directory": str(directory),
            "file_count": len(pdf_files),
            "recursive": recursive,
        }},
    )

    papers: list[StructuredPaper] = []
    for pdf_path in pdf_files:
        try:
            paper = ingest_paper(pdf_path, db=db)
            papers.append(paper)
        except (PDFParseError, IngestionError) as e:
            logger.warning(
                "Failed to ingest paper, skipping",
                extra={"extra_data": {
                    "file": str(pdf_path),
                    "error": str(e),
                }},
            )

    logger.info(
        "Batch ingestion complete",
        extra={"extra_data": {
            "total": len(pdf_files),
            "success": len(papers),
            "failed": len(pdf_files) - len(papers),
        }},
    )

    return papers


def save_paper_to_db(paper: StructuredPaper, db: Session) -> Paper:
    """Persist a StructuredPaper into PostgreSQL database tables."""
    paper_id = paper.metadata.paper_id

    # Check if paper already exists
    existing_paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if existing_paper:
        logger.info(f"Paper {paper_id} already exists in database. Updating...")
        db.delete(existing_paper)
        db.flush()

    # Create Paper record
    db_paper = Paper(
        id=paper_id,
        title=paper.metadata.title or "Untitled Document",
        abstract=paper.metadata.abstract,
        year=paper.metadata.year,
        source="PDF Ingestion",
        url=None,
    )

    # Process Authors
    if paper.metadata.authors:
        for author_name in paper.metadata.authors:
            if not author_name or not author_name.strip():
                continue
            author_name_clean = author_name.strip()
            author = db.query(Author).filter(Author.name == author_name_clean).first()
            if not author:
                author = Author(name=author_name_clean)
                db.add(author)
                db.flush()
            db_paper.authors.append(author)

    # Process Sections
    if hasattr(paper, "sections") and paper.sections:
        for idx, sec in enumerate(paper.sections):
            section_id = f"{paper_id}_sec_{idx}"
            db_section = Section(
                id=section_id,
                paper_id=paper_id,
                heading=getattr(sec, "heading", f"Section {idx+1}"),
                level=getattr(sec, "level", 1),
                order_index=idx,
            )
            db_paper.sections.append(db_section)

    # Process Chunks
    if hasattr(paper, "chunks") and paper.chunks:
        for idx, chunk in enumerate(paper.chunks):
            chunk_id = getattr(chunk, "id", f"{paper_id}_chunk_{idx}")
            db_chunk = Chunk(
                id=chunk_id,
                paper_id=paper_id,
                section=getattr(chunk, "section", None),
                subsection=getattr(chunk, "subsection", None),
                page_start=getattr(chunk, "page_start", None),
                page_end=getattr(chunk, "page_end", None),
                text=getattr(chunk, "text", str(chunk)),
                token_count=getattr(chunk, "token_count", len(str(chunk).split())),
            )
            db_paper.chunks.append(db_chunk)

    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)
    return db_paper


def save_paper(paper: StructuredPaper, output_dir: str | Path) -> str:
    """Serialize a structured paper to JSON for inspection/caching."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    filepath = output / f"{paper.metadata.paper_id}.json"

    data = paper.model_dump()
    data["_metadata"] = {
        "page_count": paper.page_count,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(filepath)


def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()