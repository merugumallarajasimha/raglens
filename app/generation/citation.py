"""Citation engine — generates stable citation IDs from backend metadata.

Citations are generated deterministically from the source chunk metadata,
not from LLM output. This prevents hallucinated or fabricated citations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.generation.context_builder import SourceReference

logger = logging.getLogger("raglens.generation.citation")


@dataclass
class Citation:
    """A citation mapping a stable ID to source metadata."""

    citation_id: int
    chunk_id: str
    paper_id: str
    title: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    evidence_snippet: str = ""
    score: float = 0.0


@dataclass
class CitationMap:
    """Maps citation IDs to source metadata for validation."""

    citations: list[Citation] = field(default_factory=list)

    def get_by_id(self, citation_id: int) -> Optional[Citation]:
        """Look up a citation by its ID."""
        for c in self.citations:
            if c.citation_id == citation_id:
                return c
        return None

    def get_by_chunk_id(self, chunk_id: str) -> Optional[Citation]:
        """Look up a citation by chunk ID."""
        for c in self.citations:
            if c.chunk_id == chunk_id:
                return c
        return None

    @property
    def max_id(self) -> int:
        """Return the maximum citation ID."""
        return max((c.citation_id for c in self.citations), default=0)

    def all_chunk_ids(self) -> set[str]:
        """Return the set of all chunk IDs in the citation map."""
        return {c.chunk_id for c in self.citations}

    def all_paper_ids(self) -> set[str]:
        """Return the set of all paper IDs in the citation map."""
        return {c.paper_id for c in self.citations}


class CitationEngine:
    """Generates and validates citation references from retrieval results.

    The engine assigns stable, sequential citation IDs [1], [2], [3], etc.
    to each source chunk, then provides lookup methods for validation.
    """

    def __init__(self) -> None:
        self._citation_map: Optional[CitationMap] = None

    def generate_citations(self, sources: list[SourceReference]) -> CitationMap:
        """Generate citation IDs for a list of sources.

        Args:
            sources: List of SourceReference objects from the context builder.

        Returns:
            A CitationMap mapping citation IDs to source metadata.
        """
        citations: list[Citation] = []

        for source in sources:
            citation = Citation(
                citation_id=source.citation_id,
                chunk_id=source.chunk_id,
                paper_id=source.paper_id,
                title=source.title,
                section=source.section,
                subsection=source.subsection,
                page_start=source.page_start,
                page_end=source.page_end,
                evidence_snippet=source.text[:200],
                score=source.score,
            )
            citations.append(citation)

        self._citation_map = CitationMap(citations=citations)

        logger.info(
            "Citations generated",
            extra={"extra_data": {"count": len(citations)}},
        )

        return self._citation_map

    def get_citation_map(self) -> Optional[CitationMap]:
        """Return the current citation map, or None if none generated."""
        return self._citation_map

    def validate_citation_reference(self, citation_id: int) -> bool:
        """Check if a citation ID exists in the current map."""
        if self._citation_map is None:
            return False
        return self._citation_map.get_by_id(citation_id) is not None


class CitationValidator:
    """Validates LLM-generated citations against the backend citation map.

    Checks that every citation referenced in the LLM output:
    - Exists in the citation map
    - Maps to a retrieved evidence chunk
    - Has valid source metadata

    Any invalid citations are flagged for removal.
    """

    def __init__(self) -> None:
        pass

    def extract_citation_ids(self, text: str) -> list[int]:
        """Extract citation IDs from text like [1], [2], [1][2], etc.

        Handles formats: [1], [2], [1][2], [1, 2], [1-3]
        """
        ids: list[int] = []
        seen: set[int] = set()

        # Match [N] patterns
        for match in re.finditer(r"\[(\d+)\]", text):
            cid = int(match.group(1))
            if cid not in seen:
                ids.append(cid)
                seen.add(cid)

        # Also match [N, M] and [N-M] formats
        for match in re.finditer(r"\[(\d+)\s*[-,]\s*(\d+)\]", text):
            start = int(match.group(1))
            end = int(match.group(2))
            for i in range(start, end + 1):
                if i not in seen:
                    ids.append(i)
                    seen.add(i)

        return sorted(ids)

    def validate(
        self,
        answer_text: str,
        citation_map: CitationMap,
    ) -> dict:
        """Validate citations in the answer text against the citation map.

        Args:
            answer_text: The LLM-generated answer text.
            citation_map: The CitationMap from the CitationEngine.

        Returns:
            Dict with:
            - 'valid': list of valid citation IDs
            - 'invalid': list of invalid (fabricated) citation IDs
            - 'missing': list of citation IDs that were valid but not referenced
            - 'is_valid': bool indicating if all citations are valid
        """
        import re

        cited_ids = self.extract_citation_ids(answer_text)
        valid_ids: list[int] = []
        invalid_ids: list[int] = []

        for cid in cited_ids:
            citation = citation_map.get_by_id(cid)
            if citation is not None:
                valid_ids.append(cid)
            else:
                invalid_ids.append(cid)

        # Citations that exist but were never referenced
        all_valid_ids = set(c.citation_id for c in citation_map.citations)
        referenced_ids = set(cited_ids)
        missing = sorted(all_valid_ids - referenced_ids)

        is_valid = len(invalid_ids) == 0

        logger.info(
            "Citation validation",
            extra={"extra_data": {
                "cited": len(cited_ids),
                "valid": len(valid_ids),
                "invalid": len(invalid_ids),
                "is_valid": is_valid,
            }},
        )

        return {
            "valid": valid_ids,
            "invalid": invalid_ids,
            "missing": missing,
            "is_valid": is_valid,
        }

    def validate_answer(
        self,
        answer_text: str,
        citation_map: CitationMap,
    ) -> tuple[str, list[Citation]]:
        """Validate and sanitize an answer, removing invalid citations.

        Args:
            answer_text: The LLM-generated answer text.
            citation_map: The CitationMap.

        Returns:
            Tuple of (sanitized_answer_text, list_of_valid_citations).
        """
        validation = self.validate(answer_text, citation_map)

        # Remove invalid citation references from the text
        sanitized = answer_text
        for invalid_id in validation["invalid"]:
            # Remove [N] references to invalid citations
            sanitized = re.sub(rf"\[{invalid_id}\]", "", sanitized)

        # Clean up any double spaces left behind
        sanitized = re.sub(r"  +", " ", sanitized).strip()

        valid_citations: list[Citation] = []
        for cid in validation["valid"]:
            citation = citation_map.get_by_id(cid)
            if citation:
                valid_citations.append(citation)

        return sanitized, valid_citations
