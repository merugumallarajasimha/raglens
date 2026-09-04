"""Metadata extraction from structured PDF content.

Extracts title, authors, abstract, and publication year by analysing
the first few pages of the parsed PDF.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.ingestion.models import StructuredPaper

logger = logging.getLogger("raglens.ingestion")


def extract_title(paper: StructuredPaper) -> str:
    """Extract the paper title.

    Priority: PDF metadata title → largest font text on page 0 → filename.
    """
    # Try PDF metadata first
    pdf_meta = getattr(paper, "_pdf_meta", {})
    if pdf_meta.get("title"):
        return pdf_meta["title"].strip()

    # Fall back to filename
    return paper.metadata.title or "Untitled"


def extract_authors(paper: StructuredPaper) -> list[str]:
    """Extract author names.

    Priority: PDF metadata author → pattern matching on first page.
    """
    authors: list[str] = []

    # Try PDF metadata
    pdf_meta = getattr(paper, "_pdf_meta", {})
    if pdf_meta.get("author"):
        # PDF authors are often comma-separated or semicolon-separated
        raw_authors = pdf_meta["author"]
        authors = [a.strip() for a in re.split(r"[,;]", raw_authors) if a.strip()]
        if authors:
            return authors

    # Try to extract from first page text using patterns
    # Author names typically appear below the title and above abstract
    pages = getattr(paper, "_pages", [])
    if pages:
        first_page_lines = [line.text for line in pages[0].lines if line.text]

        # Look for patterns like "by Author1, Author2" or just names
        # after the title line
        for line in first_page_lines:
            stripped = line.strip()
            # Skip obvious non-author lines
            if not stripped or len(stripped) < 3:
                continue
            # Author names are usually shorter and don't start with section headers
            if stripped.lower().startswith(("abstract", "introduction", "1", "i", "the ", "this ")):
                continue
            # Check if it looks like a name (contains spaces, no full sentences)
            words = stripped.split()
            if 2 <= len(words) <= 8 and not stripped.endswith("."):
                # This might be an author name
                if re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z.]+)*$", stripped) or \
                   re.match(r"^[A-Z]\.[A-Z]\.", stripped):
                    authors.append(stripped)

    return authors


def extract_abstract(paper: StructuredPaper) -> Optional[str]:
    """Extract the abstract text.

    Looks for an 'Abstract' heading and extracts text following it
    until the next section heading.
    """
    pages = getattr(paper, "_pages", [])
    if pages:
        # Search first 3 pages for abstract
        for page_idx in range(min(3, len(pages))):
            lines = pages[page_idx].lines
            for i, line in enumerate(lines):
                text = line.text.strip().lower()
                if text in ("abstract", "abstract —", "abstract—", "abstract:"):
                    # Extract text following the abstract heading
                    abstract_parts: list[str] = []
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j]
                        next_text = next_line.text.strip()
                        # Check if we've hit the next section
                        next_lower = next_text.lower()
                        if _is_section_heading(next_lower):
                            break
                        if next_text:
                            abstract_parts.append(next_text)
                    if abstract_parts:
                        return " ".join(abstract_parts)

    # Fallback: look in raw_text for "Abstract" keyword
    if paper.raw_text:
        # Find "Abstract" as a standalone heading followed by content
        match = re.search(
            r"(?:^|\n)\s*[Aa]bstract\s*[—–-]?\s*\n+",
            paper.raw_text,
        )
        if match:
            start = match.end()
            rest = paper.raw_text[start:]
            # Find the next section heading
            for pattern in [
                r"\n\n\d+\.?\s+[A-Z]",
                r"\n\nINTRODUCTION",
                r"\n\n1\s+",
                r"\n\n\n[0-9IVX]",
            ]:
                section_match = re.search(pattern, rest, re.IGNORECASE)
                if section_match:
                    return rest[:section_match.start()].strip()
            return rest[:500].strip()

    return None


def extract_year(paper: StructuredPaper) -> Optional[int]:
    """Extract publication year from metadata or content.

    Looks for: PDF metadata dates, 4-digit year in first page content.
    """
    # Try PDF metadata creation date (format: "D:20240115120000+00'00'")
    pdf_meta = getattr(paper, "_pdf_meta", {})
    for date_key in ("creation_date", "mod_date"):
        date_str = pdf_meta.get(date_key, "")
        if date_str:
            year_match = re.search(r"(\d{4})", date_str)
            if year_match:
                year = int(year_match.group(1))
                if 1990 <= year <= 2030:
                    return year

    # Look for 4-digit year in first page
    pages = getattr(paper, "_pages", [])
    if pages:
        first_page_text = pages[0].text
        # Look for years in typical positions (footer of first page)
        for match in re.finditer(r"\b(20\d{2})\b", first_page_text):
            year = int(match.group(1))
            if 2010 <= year <= 2030:
                return year

    return None


def extract_doi(paper: StructuredPaper) -> Optional[str]:
    """Extract DOI from PDF metadata or content."""
    # Try metadata
    pdf_meta = getattr(paper, "_pdf_meta", {})
    for key in ("subject", "keywords"):
        val = pdf_meta.get(key, "")
        if val:
            doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", val, re.IGNORECASE)
            if doi_match:
                return doi_match.group(0)

    # Look in raw text
    if paper.raw_text:
        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", paper.raw_text, re.IGNORECASE)
        if doi_match:
            return doi_match.group(0)

    return None


def extract_references(paper: StructuredPaper) -> list[str]:
    """Extract reference list from the paper.

    Detects the references section and extracts individual reference entries.
    """
    pages = getattr(paper, "_pages", [])
    if not pages:
        return []

    references: list[str] = []
    in_references = False

    for page in pages:
        for line in page.lines:
            text = line.text.strip()
            if not text:
                continue

            text_lower = text.lower()

            # Detect references section start
            if not in_references:
                if text_lower in ("references", "references and notes", "bibliography",
                                  "works cited", "references:") or \
                   re.match(r"^references?\s*[:\-]?\s*$", text_lower):
                    in_references = True
                    continue
                continue

            # Detect end of references (appendices, etc.)
            if re.match(r"^(appendix|acknowledg|conclusion)", text_lower):
                if len(references) > 0:
                    break
                in_references = False
                continue

            # Collect reference entries
            # References often start with a number or author name
            if re.match(r"^(\d+|[\d\.]+)\s+", text) or re.match(r"^[A-Z][a-z]+,\s", text):
                references.append(text)
            elif references and (
                re.match(r"^\[?\d+\]?\.?\s+[A-Z]", text) or
                len(references[-1].split()) > 5  # multi-line continuation
            ):
                # Continuation of previous reference
                references[-1] += " " + text
            elif len(references) > 0 and not text_lower.startswith(("abstract", "introduction")):
                references[-1] += " " + text

    return references


def extract_metadata(paper: StructuredPaper) -> StructuredPaper:
    """Fill in all metadata fields on the structured paper.

    This is the main entry point that the ingestion pipeline calls.

    Args:
        paper: A partially-parsed paper from parse_pdf().

    Returns:
        The same paper with metadata fields populated.
    """
    paper.metadata.title = extract_title(paper)
    paper.metadata.authors = extract_authors(paper)
    paper.metadata.abstract = extract_abstract(paper)
    paper.metadata.year = extract_year(paper)
    paper.metadata.doi = extract_doi(paper)
    paper.metadata.references = extract_references(paper)

    # Clean up temporary metadata attribute
    # (structure parser handles _pages cleanup)
    if hasattr(paper, "_pdf_meta"):
        delattr(paper, "_pdf_meta")

    logger.info(
        "Metadata extracted",
        extra={"extra_data": {
            "paper_id": paper.metadata.paper_id,
            "title": paper.metadata.title,
            "authors": len(paper.metadata.authors),
            "has_abstract": paper.metadata.abstract is not None,
            "year": paper.metadata.year,
        }},
    )

    return paper


# ── Helpers ──────────────────────────────────────────────────────────────

_SECTION_HEADINGS = {
    "abstract", "introduction", "background", "related work", "related work",
    "method", "methodology", "approach", "methods", "materials",
    "experiments", "experimental setup", "results", "discussion",
    "conclusion", "conclusions", "future work", "acknowledgments",
    "acknowledgements", "references", "bibliography", "appendix",
    "appendices", "supplementary material", "data availability",
}


def _is_section_heading(text: str) -> bool:
    """Check if a text line is likely a section heading."""
    text_lower = text.strip().lower()

    # Exact matches for known section names
    for heading in _SECTION_HEADINGS:
        if text_lower.startswith(heading) and (
            len(text_lower) == len(heading) or
            text_lower[len(heading):].startswith((" ", "—", "-", ":", "."))
        ):
            return True

    # Numbered sections: "1.", "1.1", "2.1.1", etc.
    if re.match(r"^[IVX]+\.\s+", text) or re.match(r"^\d+(\.\d+)*\s+", text):
        # But not if it's a long sentence
        if len(text.split()) <= 12:
            return True

    return False
