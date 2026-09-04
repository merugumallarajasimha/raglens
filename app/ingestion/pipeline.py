"""Ingestion pipeline — orchestrates PDF parsing, structure parsing, and metadata extraction.

This module provides the high-level API for ingesting a single PDF or a
directory of PDFs into structured paper objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from app.core.exceptions import IngestionError, PDFParseError
from app.ingestion.metadata_extractor import extract_metadata
from app.ingestion.models import IngestResult, StructuredPaper
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.structure_parser import parse_structure

logger = logging.getLogger("raglens.ingestion")


def ingest_paper(file_path: str | Path) -> StructuredPaper:
    """Ingest a single PDF and return a fully structured paper.

    Pipeline:
        1. parse_pdf()     — extract text + font metadata → StructuredPaper
        2. parse_structure()  — detect sections/subsections
        3. extract_metadata()  — extract title, authors, abstract, year

    Args:
        file_path: Path to the PDF file.

    Returns:
        A StructuredPaper with sections and metadata populated.

    Raises:
        PDFParseError: If the PDF cannot be parsed.
        IngestionError: If ingestion fails for other reasons.
    """
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

    # Step 2: Extract metadata (uses _pdf_meta and _pages, then cleans _pdf_meta)
    paper = extract_metadata(paper)

    # Step 3: Parse structure (uses _pages, then cleans up _pages)
    paper = parse_structure(paper)

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
) -> list[StructuredPaper]:
    """Ingest all PDFs from a directory.

    Args:
        dir_path: Path to the directory containing PDFs.
        recursive: If True, search subdirectories recursively.

    Returns:
        List of StructuredPaper objects. Failed files are skipped with a warning.
    """
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
            paper = ingest_paper(pdf_path)
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


def save_paper(paper: StructuredPaper, output_dir: str | Path) -> str:
    """Serialize a structured paper to JSON for inspection/caching.

    Args:
        paper: The structured paper to save.
        output_dir: Directory to save the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    filepath = output / f"{paper.metadata.paper_id}.json"

    data = paper.model_dump()
    data["_metadata"] = {
        "page_count": paper.page_count,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(
        "Paper saved to JSON",
        extra={"extra_data": {
            "paper_id": paper.metadata.paper_id,
            "path": str(filepath),
        }},
    )

    return str(filepath)


def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
