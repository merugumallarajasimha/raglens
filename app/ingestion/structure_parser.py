"""Structure parser — detects section/subsection hierarchy from PDF text.

Uses font-size analysis, bold detection, and pattern matching to identify
section headings and group paragraphs into a hierarchical structure.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.ingestion.models import Paragraph, Section, StructuredPaper
from app.ingestion.metadata_extractor import _is_section_heading

logger = logging.getLogger("raglens.ingestion")


def _compute_body_font_size(lines: list, max_pages: int = 3) -> float:
    """Find the most common font size across pages, used as body text size."""
    from collections import Counter

    size_counts: Counter = Counter()
    for page in lines[:max_pages]:
        for line in page.lines:
            if line.spans:
                # Round to handle minor float differences
                size_counts[round(line.max_font_size, 1)] += 1

    if size_counts:
        return size_counts.most_common(1)[0][0]
    return 11.0  # fallback


def _classify_heading_level(font_size: float, body_size: float, is_bold: bool,
                             text: str) -> Optional[int]:
    """Classify a line as a heading and return its level (1, 2, 3) or None.

    Uses font-size ratios, bold flag, and pattern matching.
    """
    ratio = font_size / body_size if body_size > 0 else 1.0
    text_stripped = text.strip()

    # Always check for section number patterns first (most reliable)
    # e.g., "1 Introduction", "1.1 Background", "2.1.1 Detail"
    num_match = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", text_stripped)
    if num_match:
        num_part = num_match.group(1)
        dots = num_part.count(".")
        return min(dots + 1, 3)  # 0 dots -> level 1, 1 dot -> level 2, etc.

    # Roman numeral patterns
    roman_match = re.match(r"^([IVX]+)\.?\s+(.+)$", text_stripped)
    if roman_match:
        return 1

    # Letter-based patterns: "A. Introduction", "Appendix A"
    letter_match = re.match(r"^([A-Z])\.\s+(.+)$", text_stripped)
    if letter_match:
        return 1

    # Font-size based classification
    if ratio >= 1.35:
        return 1
    elif ratio >= 1.10:
        return 2
    elif is_bold and ratio >= 1.0 and len(text_stripped) < 80:
        # Bold text slightly larger than body could be a heading
        return 3

    return None


def _extract_headings(pages: list) -> list[dict]:
    """Extract all heading lines from the paper's pages.

    Returns a list of dicts with keys: text, level, page, y_center.
    """
    body_size = _compute_body_font_size(pages)

    headings: list[dict] = []

    for page in pages:
        for line in page.lines:
            if not line.spans or not line.text.strip():
                continue

            text = line.text
            font_size = line.max_font_size
            is_bold = line.is_bold

            level = _classify_heading_level(font_size, body_size, is_bold, text)

            if level is not None:
                headings.append({
                    "text": text.strip(),
                    "level": level,
                    "page": line.page,
                    "y_center": line.y_center,
                })

    return headings


def _build_section_hierarchy(headings: list[dict], pages: list) -> list[Section]:
    """Build a hierarchical section structure from detected headings.

    Groups paragraphs that appear between headings into sections.
    Falls back to treating all content as one section if no headings found.
    """
    if not headings:
        return _fallback_no_headings(pages)

    sections: list[Section] = []
    current_top: Section | None = None
    current_sub: list[Section] = []
    current_level2: Section | None = None

    for i, heading in enumerate(headings):
        level = heading["level"]
        text = heading["text"]
        page_start = heading["page"]

        # Collect paragraphs between this heading and the next
        paragraph_text = _collect_paragraphs_between(
            pages, heading, headings[i + 1] if i + 1 < len(headings) else None
        )

        if level == 1:
            # Close any open level-2
            if current_level2 is not None and current_top is not None:
                current_top.subsections.extend(current_sub)
                current_sub = []
                current_level2 = None
            # Close current top-level
            if current_top is not None:
                sections.append(current_top)

            current_top = Section(
                heading=text,
                level=1,
                page_start=page_start,
                paragraphs=paragraph_text,
            )
            current_sub = []
            current_level2 = None

        elif level == 2:
            sec = Section(
                heading=text,
                level=2,
                page_start=page_start,
                paragraphs=paragraph_text,
            )
            if current_top is not None:
                current_top.subsections.append(sec)
                current_level2 = sec
                current_sub = []
            else:
                # Orphan subsection — treat as top-level
                sections.append(sec)

        elif level == 3:
            sec = Section(
                heading=text,
                level=3,
                page_start=page_start,
                paragraphs=paragraph_text,
            )
            if current_level2 is not None:
                current_level2.subsections.append(sec)
                current_sub.append(sec)
            elif current_top is not None:
                current_top.subsections.append(sec)
            else:
                sections.append(sec)

    # Close remaining sections
    if current_top is not None:
        if current_level2 is not None and current_sub:
            current_level2.subsections.extend(current_sub)
        sections.append(current_top)

    # If we only got level-2 sections with no parent, wrap them
    if sections and all(s.level >= 2 for s in sections):
        wrapper = Section(heading="Paper Body", level=1, page_start=0)
        wrapper.subsections = sections
        return [wrapper]

    return sections


def _collect_paragraphs_between(
    pages: list,
    start_heading: dict,
    end_heading: Optional[dict],
) -> list[Paragraph]:
    """Collect paragraph text between two heading markers."""
    paragraphs: list[Paragraph] = []

    start_page = start_heading["page"]
    start_y = start_heading["y_center"]

    end_page: int | None = None
    end_y: float | None = None
    if end_heading is not None:
        end_page = end_heading["page"]
        end_y = end_heading["y_center"]

    for page_idx in range(start_page, len(pages)):
        page = pages[page_idx]

        if end_page is not None and page_idx > end_page:
            break

    collecting = True
    last_y: float | None = None
    for line in page.lines:
        if not line.text.strip():
            continue

        # Skip lines that are themselves headings
        if _is_heading_line(line):
            continue

        # Check if we've reached the end heading
        if end_page is not None and page_idx == end_page:
            if line.y_center >= end_y:
                collecting = False
                break

        # Skip if on the same page as the start heading and above it
        if page_idx == start_page and line.y_center < start_y:
            continue

        if collecting:
            # Group consecutive lines into paragraphs
            if last_y is not None and abs(line.y_center - last_y) > 10:
                # New paragraph
                paragraphs.append(Paragraph(text=line.text, page=page_idx))
            elif last_y is not None:
                # Continuation of current paragraph
                paragraphs[-1].text += "\n" + line.text
            else:
                # First paragraph
                paragraphs.append(Paragraph(text=line.text, page=page_idx))
            last_y = line.y_center

    return paragraphs


def _is_heading_line(line) -> bool:
    """Quick check: is this line a heading that should be skipped in paragraphs?"""
    text = line.text.strip()
    if not text:
        return False
    # Short, potentially bold, numbered or known heading
    if re.match(r"^\d+(\.\d+)*\s+", text) and len(text.split()) <= 12:
        return True
    if re.match(r"^[IVX]+\.?\s+", text) and len(text.split()) <= 12:
        return True
    for heading in ("abstract", "introduction", "background", "method",
                     "methodology", "experiments", "results", "conclusion",
                     "references", "appendix", "acknowledgments", "acknowledgements",
                     "related work", "future work", "discussion"):
        if text.lower() == heading or text.lower().startswith(heading + " "):
            return True
    return False


def _fallback_no_headings(pages: list) -> list[Section]:
    """When no headings are detected, wrap all content in a single section."""
    if not pages:
        return []

    paragraphs: list[Paragraph] = []
    for page in pages:
        current: Paragraph | None = None
        for line in page.lines:
            if not line.text.strip():
                current = None
                continue
            if current is None:
                current = Paragraph(text=line.text, page=page.page_no)
                paragraphs.append(current)
            elif abs(line.y_center - current.y_center) > 10:
                current = Paragraph(text=line.text, page=page.page_no)
                paragraphs.append(current)
            else:
                current.text += "\n" + line.text

    # Use the first non-empty line as the section title
    first_text = ""
    for p in paragraphs:
        if p.text.strip():
            first_text = p.text.strip().split("\n")[0]
            break

    return [Section(
        heading=first_text or "Body",
        level=1,
        page_start=pages[0].page_no,
        paragraphs=paragraphs,
    )]


def parse_structure(paper: StructuredPaper) -> StructuredPaper:
    """Parse the section structure of a paper from its pages.

    This must be called after parse_pdf() and before extract_metadata()
    (since it consumes the _pages attribute).

    Args:
        paper: A paper with _pages attribute from parse_pdf().

    Returns:
        The paper with sections populated.
    """
    pages = getattr(paper, "_pages", [])
    if not pages:
        logger.warning(
            "No pages found for structure parsing",
            extra={"extra_data": {"paper_id": paper.metadata.paper_id}},
        )
        return paper

    headings = _extract_headings(pages)
    logger.info(
        "Headings detected",
        extra={"extra_data": {
            "paper_id": paper.metadata.paper_id,
            "heading_count": len(headings),
        }},
    )

    if headings:
        paper.sections = _build_section_hierarchy(headings, pages)
    else:
        paper.sections = _fallback_no_headings(pages)

    # Clean up temporary pages attribute
    if hasattr(paper, "_pages"):
        delattr(paper, "_pages")

    total_paragraphs = sum(
        len(s.paragraphs) + sum(len(s2.paragraphs) for s2 in s.subsections)
        for s in paper.sections
    )
    logger.info(
        "Structure parsed",
        extra={"extra_data": {
            "paper_id": paper.metadata.paper_id,
            "section_count": len(paper.sections),
            "paragraph_count": total_paragraphs,
        }},
    )

    return paper
