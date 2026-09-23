"""Structure-aware chunking for research papers.

Supports three chunking strategies:
1. Fixed-size: splits text by token count at fixed boundaries
2. Paragraph-based: groups text into paragraph-level chunks
3. Section-aware: respects section/subsection boundaries when chunking
"""

from __future__ import annotations

import hashlib
import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.ingestion.models import Paragraph, Section, StructuredPaper

logger = logging.getLogger("raglens.chunking")


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    PARAGRAPH = "paragraph"
    SECTION_AWARE = "section_aware"


class Chunk(BaseModel):
    """A single text chunk with source metadata for citations."""

    chunk_id: str
    paper_id: str
    title: str
    section: Optional[str] = Field(default=None, description="Top-level section heading")
    subsection: Optional[str] = Field(default=None, description="Subsection heading")
    page_start: Optional[int] = Field(default=None)
    page_end: Optional[int] = Field(default=None)
    token_count: int
    text: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple approximation: 1 token ≈ 4 characters (standard for English text).
    For more accurate counts, the embedding model's tokenizer is used when available.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def make_chunk_id(paper_id: str, section: str | None, subsection: str | None,
                  chunk_index: int) -> str:
    """Generate a deterministic chunk ID.

    Format: {paper_id}_{section_hash}_{subsection_hash}_{chunk_index}
    """
    sec_part = _short_hash(section) if section else "root"
    sub_part = _short_hash(subsection) if subsection else "nosub"
    return f"{paper_id}_{sec_part}_{sub_part}_{chunk_index:04d}"


def _short_hash(text: str | None, length: int = 8) -> str:
    """Generate a short deterministic hash from text."""
    if not text:
        return "none"
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


class Chunker:
    """Structure-aware chunker for research papers.

    Args:
        chunk_size: Target number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
        strategy: Chunking strategy to use.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        strategy: str | ChunkingStrategy = ChunkingStrategy.SECTION_AWARE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = ChunkingStrategy(strategy)

    def chunk_paper(self, paper: StructuredPaper) -> list[Chunk]:
        """Chunk a structured paper into text chunks.

        Args:
            paper: A StructuredPaper with sections parsed.

        Returns:
            List of Chunk objects with source metadata.
        """
        if self.strategy == ChunkingStrategy.SECTION_AWARE:
            return self._chunk_section_aware(paper)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraph(paper)
        elif self.strategy == ChunkingStrategy.FIXED:
            return self._chunk_fixed(paper)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _chunk_section_aware(self, paper: StructuredPaper) -> list[Chunk]:
        """Chunk respecting section and subsection boundaries.

        Paragraphs within a section are grouped into chunks up to chunk_size tokens.
        When a section boundary is crossed, the current chunk is finalized with
        overlap from the previous chunks in the same section.
        """
        chunks: list[Chunk] = []

        for section in paper.sections:
            section_chunks, _ = self._chunk_section_recursive(
                section, paper.metadata.paper_id, paper.metadata.title,
                parent_heading=None,
            )
            chunks.extend(section_chunks)

        logger.info(
            "Section-aware chunking complete",
            extra={"extra_data": {
                "paper_id": paper.metadata.paper_id,
                "chunk_count": len(chunks),
                "strategy": self.strategy.value,
            }},
        )

        return chunks

    def _chunk_section_recursive(
        self,
        section: Section,
        paper_id: str,
        title: str,
        parent_heading: str | None,
        chunk_index: int = 0,
    ) -> tuple[list[Chunk], int]:
        """Recursively chunk a section and its subsections.

        Returns (chunks, next_chunk_index).
        """
        chunks: list[Chunk] = []
        section_heading = section.heading

        # Chunk paragraphs in this section
        section_chunks, chunk_index = self._chunk_paragraphs(
            paragraphs=section.paragraphs,
            paper_id=paper_id,
            title=title,
            section_heading=section_heading,
            subsection_heading=parent_heading,
            start_index=chunk_index,
        )
        chunks.extend(section_chunks)

        # Recursively process subsections
        for subsection in section.subsections:
            sub_chunks, chunk_index = self._chunk_section_recursive(
                subsection, paper_id, title,
                parent_heading=section_heading,
                chunk_index=chunk_index,
            )
            chunks.extend(sub_chunks)

        return chunks, chunk_index

    def _chunk_paragraphs(
        self,
        paragraphs: list,
        paper_id: str,
        title: str,
        section_heading: str | None,
        subsection_heading: str | None,
        start_index: int,
    ) -> tuple[list[Chunk], int]:
        """Chunk a list of paragraphs into chunks of approximately chunk_size tokens.

        Uses overlap between consecutive chunks.
        """
        chunks: list[Chunk] = []
        if not paragraphs:
            return chunks, start_index

        current_text: list[str] = []
        current_tokens = 0
        chunk_index = start_index
        page_start: int | None = None
        page_end: int | None = None

        for para in paragraphs:
            para_tokens = estimate_tokens(para.text)

            # If adding this paragraph would exceed chunk_size and we have content,
            # finalize the current chunk
            if current_tokens + para_tokens > self.chunk_size and current_text:
                chunk, overlap_parts, overlap_token_count = self._finalize_chunk(
                    current_text, paper_id, title, section_heading,
                    subsection_heading, chunk_index, page_start, page_end,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap
                current_text = overlap_parts
                current_tokens = overlap_token_count
                page_start = page_end  # carry forward
                page_end = None

            current_text.append(para.text)
            current_tokens += para_tokens
            if page_start is None:
                page_start = para.page
            page_end = para.page

        # Finalize remaining text
        if current_text:
            chunk, _, _ = self._finalize_chunk(
                current_text, paper_id, title, section_heading,
                subsection_heading, chunk_index, page_start, page_end,
            )
            chunks.append(chunk)
            chunk_index += 1

        return chunks, chunk_index

    def _finalize_chunk(
        self,
        text_parts: list[str],
        paper_id: str,
        title: str,
        section: str | None,
        subsection: str | None,
        chunk_index: int,
        page_start: int | None,
        page_end: int | None,
    ) -> tuple[Chunk, list[str], int]:
        """Finalize a chunk from accumulated text parts.

        Returns (chunk, overlap_text_parts, overlap_token_count).
        """
        text = "\n".join(text_parts)

        # Prepend section header for context (improves retrieval for section-specific queries)
        header_parts = []
        if section:
            header_parts.append(f"Section: {section}")
        if subsection:
            header_parts.append(f"Subsection: {subsection}")
        if header_parts:
            text = " | ".join(header_parts) + "\n" + text

        # Compute overlap text (last N tokens worth)
        overlap_parts: list[str] = []
        overlap_tokens = 0
        for part in reversed(text_parts):
            t = estimate_tokens(part)
            if overlap_tokens + t <= self.chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_tokens += t
            else:
                break

        chunk_id = make_chunk_id(paper_id, section, subsection, chunk_index)

        chunk = Chunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            title=title,
            section=section,
            subsection=subsection,
            page_start=page_start,
            page_end=page_end,
            token_count=estimate_tokens(text),
            text=text,
            metadata={
                "section_level": _get_section_level(section, subsection),
                "overlap_tokens": overlap_tokens,
            },
        )

        return chunk, overlap_parts, overlap_tokens

    def _chunk_paragraph(self, paper: StructuredPaper) -> list[Chunk]:
        """Chunk based purely on paragraph boundaries (no section awareness)."""
        chunks: list[Chunk] = []
        chunk_index = 0

        all_paragraphs: list[tuple] = []
        for section in paper.sections:
            for para in section.paragraphs:
                all_paragraphs.append((para, section.heading, None))
            for sub in section.subsections:
                for para in sub.paragraphs:
                    all_paragraphs.append((para, section.heading, sub.heading))

        current_text: list[str] = []
        current_tokens = 0
        page_start: int | None = None
        page_end: int | None = None
        current_section: str | None = None
        current_subsection: str | None = None

        for para, section_heading, subsection_heading in all_paragraphs:
            para_tokens = estimate_tokens(para.text)

            if current_tokens + para_tokens > self.chunk_size and current_text:
                text = "\n".join(current_text)
                # Prepend section header for context
                header_parts = []
                if current_section:
                    header_parts.append(f"Section: {current_section}")
                if current_subsection:
                    header_parts.append(f"Subsection: {current_subsection}")
                if header_parts:
                    text = " | ".join(header_parts) + "\n" + text

                chunk = Chunk(
                    chunk_id=make_chunk_id(paper.metadata.paper_id, current_section,
                                           current_subsection, chunk_index),
                    paper_id=paper.metadata.paper_id,
                    title=paper.metadata.title,
                    section=current_section,
                    subsection=current_subsection,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=current_tokens,
                    text=text,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Reset with overlap
                current_text = [para.text]
                current_tokens = para_tokens
                page_start = para.page
                page_end = para.page
                current_section = section_heading
                current_subsection = subsection_heading
            else:
                if not current_text:
                    page_start = para.page
                    current_section = section_heading
                    current_subsection = subsection_heading
                current_text.append(para.text)
                current_tokens += para_tokens
                page_end = para.page

        if current_text:
            text = "\n".join(current_text)
            # Prepend section header for context
            header_parts = []
            if current_section:
                header_parts.append(f"Section: {current_section}")
            if current_subsection:
                header_parts.append(f"Subsection: {current_subsection}")
            if header_parts:
                text = " | ".join(header_parts) + "\n" + text

            chunk = Chunk(
                chunk_id=make_chunk_id(paper.metadata.paper_id, current_section,
                                       current_subsection, chunk_index),
                paper_id=paper.metadata.paper_id,
                title=paper.metadata.title,
                section=current_section,
                subsection=current_subsection,
                page_start=page_start,
                page_end=page_end,
                token_count=current_tokens,
                text=text,
            )
            chunks.append(chunk)

        logger.info(
            "Paragraph chunking complete",
            extra={"extra_data": {
                "paper_id": paper.metadata.paper_id,
                "chunk_count": len(chunks),
            }},
        )
        return chunks

    def _chunk_fixed(self, paper: StructuredPaper) -> list[Chunk]:
        """Fixed-size chunking: splits all text into fixed token windows."""
        chunks: list[Chunk] = []
        chunk_index = 0

        # Flatten all text with page tracking
        all_text = paper.raw_text or paper.metadata.abstract or ""
        if not all_text:
            for section in paper.sections:
                all_text += section.heading + "\n" + "\n".join(
                    p.text for p in section.paragraphs
                )

        if not all_text:
            return chunks

        words = all_text.split()
        words_per_chunk = max(1, self.chunk_size // 4)  # rough estimate
        overlap_words = max(1, self.chunk_overlap // 4)

        i = 0
        while i < len(words):
            chunk_words = words[i:i + words_per_chunk]
            text = " ".join(chunk_words)

            # Estimate page for this chunk
            page_start, page_end = _estimate_page_range(text, paper)

            chunk = Chunk(
                chunk_id=f"{paper.metadata.paper_id}_fixed_{chunk_index:04d}",
                paper_id=paper.metadata.paper_id,
                title=paper.metadata.title,
                section=None,
                subsection=None,
                page_start=page_start,
                page_end=page_end,
                token_count=estimate_tokens(text),
                text=text,
            )
            chunks.append(chunk)
            chunk_index += 1
            i += words_per_chunk - overlap_words

        logger.info(
            "Fixed-size chunking complete",
            extra={"extra_data": {
                "paper_id": paper.metadata.paper_id,
                "chunk_count": len(chunks),
            }},
        )
        return chunks


def _get_section_level(section: str | None, subsection: str | None) -> str:
    """Return the section level description."""
    if section and subsection:
        return "subsection"
    elif section:
        return "section"
    return "body"


def _estimate_page_range(text: str, paper: StructuredPaper) -> tuple[int, int]:
    """Estimate page range for fixed-size chunks based on section page info."""
    if paper.sections:
        first_section = paper.sections[0]
        return first_section.page_start, first_section.page_start + 1
    return 0, 0


def chunk_paper(
    paper: StructuredPaper,
    settings: Settings | None = None,
) -> list[Chunk]:
    """Convenience function to chunk a paper using settings.

    Args:
        paper: The structured paper to chunk.
        settings: Application settings (chunk_size, chunk_overlap, strategy).

    Returns:
        List of Chunk objects.
    """
    if settings is None:
        settings = Settings()

    chunker = Chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        strategy=settings.chunking_strategy,
    )
    return chunker.chunk_paper(paper)
