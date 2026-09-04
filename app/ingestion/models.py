"""Data models for the ingestion pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    """A paragraph within a section."""

    text: str
    page: int = Field(..., description="0-based page number where this paragraph appears")


class Section(BaseModel):
    """A section of the paper with hierarchical subsections."""

    heading: str
    level: int = Field(..., description="1 = section, 2 = subsection, etc.")
    page_start: int
    paragraphs: list[Paragraph] = Field(default_factory=list)
    subsections: list["Section"] = Field(default_factory=list)

    def all_chunks_text(self) -> list[str]:
        """Return all paragraph texts in this section and subsections."""
        result = [p.text for p in self.paragraphs]
        for sub in self.subsections:
            result.extend(sub.all_chunks_text())
        return result


class PaperMetadata(BaseModel):
    """Metadata extracted from a paper."""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    source: str = Field(default="local")
    url: Optional[str] = None
    doi: Optional[str] = None
    references: list[str] = Field(default_factory=list)


class StructuredPaper(BaseModel):
    """A fully parsed research paper with structure."""

    metadata: PaperMetadata
    sections: list[Section] = Field(default_factory=list)
    page_count: int = 0
    raw_text: Optional[str] = None

    @property
    def title(self) -> str:
        return self.metadata.title

    @property
    def paper_id(self) -> str:
        return self.metadata.paper_id


class IngestResult(BaseModel):
    """Result of ingesting a single paper."""

    paper_id: str
    title: str
    success: bool
    page_count: int
    section_count: int
    chunk_count: int = 0
    error: Optional[str] = None
    ingested_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
