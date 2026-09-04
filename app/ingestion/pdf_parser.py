"""PDF parsing using PyMuPDF — extracts structured text with font/size information."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf

from app.core.exceptions import PDFParseError
from app.ingestion.models import StructuredPaper

logger = logging.getLogger("raglens.ingestion")


@dataclass
class TextSpan:
    """A single text span with font metadata from PyMuPDF."""

    text: str
    page_no: int
    font_size: float
    font_name: str
    is_bold: bool
    bbox_top: float
    bbox_bottom: float

    @property
    def y_center(self) -> float:
        return (self.bbox_top + self.bbox_bottom) / 2


@dataclass
class TextLine:
    """A line of text composed of one or more spans."""

    spans: list[TextSpan] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.spans).strip()

    @property
    def max_font_size(self) -> float:
        return max(s.font_size for s in self.spans) if self.spans else 0.0

    @property
    def is_bold(self) -> bool:
        return any(s.is_bold for s in self.spans)

    @property
    def page(self) -> int:
        return self.spans[0].page_no if self.spans else 0

    @property
    def y_center(self) -> float:
        return self.spans[0].y_center if self.spans else 0.0


@dataclass
class PageText:
    """All text lines on a single page."""

    page_no: int
    lines: list[TextLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)

    def sorted_lines(self) -> list[TextLine]:
        """Return lines sorted by vertical position (top to bottom)."""
        return sorted(self.lines, key=lambda l: l.y_center)


def compute_paper_id(file_path: str | Path) -> str:
    """Generate a deterministic paper ID from file content hash."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return f"paper_{hasher.hexdigest()[:16]}"


def extract_pdf_metadata(doc: pymupdf.Document) -> dict:
    """Extract raw metadata from PDF document."""
    meta = doc.metadata
    return {
        "title": meta.get("title", "").strip() or None,
        "author": meta.get("author", "").strip() or None,
        "subject": meta.get("subject", "").strip() or None,
        "keywords": meta.get("keywords", "").strip() or None,
        "creator": meta.get("creator", "").strip() or None,
        "producer": meta.get("producer", "").strip() or None,
        "creation_date": meta.get("creationDate", "").strip() or None,
        "mod_date": meta.get("modDate", "").strip() or None,
    }


def parse_pdf(file_path: str | Path) -> StructuredPaper:
    """Parse a PDF file into a structured paper representation.

    Extracts per-page text with font metadata, preserving page boundaries
    and structural information needed for section-aware chunking.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A StructuredPaper with metadata and raw page text.

    Raises:
        PDFParseError: If the PDF cannot be read or is empty.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    paper_id = compute_paper_id(path)

    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        raise PDFParseError(f"Failed to open PDF {path}: {e}", {"file": str(path)})

    page_count = doc.page_count
    if page_count == 0:
        raise PDFParseError(f"PDF has no pages: {path}", {"file": str(path)})

    logger.info(
        f"Parsing PDF",
        extra={"extra_data": {
            "paper_id": paper_id,
            "file": str(path),
            "pages": page_count,
        }},
    )

    # Extract text from each page with span-level detail
    pages: list[PageText] = []
    all_text_parts: list[str] = []

    for page_idx in range(page_count):
        page = doc.load_page(page_idx)
        page_no = page.number  # 0-based

        # get_text("dict") gives structured output with spans and font info
        page_dict = page.get_text("dict")

        page_text = PageText(page_no=page_no)

        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:  # skip non-text blocks (images, etc.)
                continue

            for line_data in block.get("lines", []):
                text_line = TextLine()
                for span_data in line_data.get("spans", []):
                    span = TextSpan(
                        text=span_data["text"],
                        page_no=page_no,
                        font_size=span_data.get("size", 0.0),
                        font_name=span_data.get("font", ""),
                        is_bold=_detect_bold(span_data.get("font", ""), span_data.get("flags", 0)),
                        bbox_top=span_data.get("bbox", [0, 0, 0, 0])[1],
                        bbox_bottom=span_data.get("bbox", [0, 0, 0, 0])[3],
                    )
                    text_line.spans.append(span)

                if text_line.text:
                    page_text.lines.append(text_line)

        # Sort lines by vertical position
        page_text.lines.sort(key=lambda l: l.y_center)
        pages.append(page_text)
        all_text_parts.append(page_text.text)

    full_text = "\n\n".join(all_text_parts)

    # Extract metadata
    pdf_meta = extract_pdf_metadata(doc)

    doc.close()

    logger.info(
        "PDF parsed successfully",
        extra={"extra_data": {
            "paper_id": paper_id,
            "pages": page_count,
            "chars": len(full_text),
        }},
    )

    # Return a StructuredPaper with metadata placeholder; structure/metadata
    # extraction is handled by metadata_extractor and structure_parser.
    from app.ingestion.models import PaperMetadata

    metadata = PaperMetadata(
        paper_id=paper_id,
        title=pdf_meta.get("title") or path.stem,
        authors=[],
        year=None,
        abstract=None,
        source="local",
        url=None,
    )

    paper = StructuredPaper(
        metadata=metadata,
        sections=[],
        page_count=page_count,
        raw_text=full_text,
    )
    paper._pdf_meta = pdf_meta  # type: ignore[attr-defined]
    paper._pages = pages  # type: ignore[attr-defined]

    return paper


def _detect_bold(font_name: str, flags: int) -> bool:
    """Detect if a span is bold based on font name or Pygments flags.

    PyMuPDF span flags: bit 4 (value 4) indicates bold.
    """
    if flags & 4:
        return True
    font_lower = font_name.lower()
    return "bold" in font_lower or "bold" in font_lower


def get_document_info(file_path: str | Path) -> dict:
    """Get basic document info without full parsing."""
    path = Path(file_path)
    doc = pymupdf.open(str(path))
    info = {
        "page_count": doc.page_count,
        "metadata": doc.metadata,
    }
    doc.close()
    return info
