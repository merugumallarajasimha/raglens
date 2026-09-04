"""Metadata filtering for retrieval results.

Provides filter utilities to narrow down search results based on
paper metadata such as paper_id, year, source, section, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.retrieval.hybrid import HybridResult
from app.database.vector_store import RetrievedChunk

logger = logging.getLogger("raglens.retrieval.filters")


@dataclass
class RetrievalFilters:
    """Structured filter specification for retrieval.

    All fields are optional; None means no filtering on that dimension.
    """

    paper_ids: Optional[list[str]] = None
    sources: Optional[list[str]] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    sections: Optional[list[str]] = None
    authors: Optional[list[str]] = None


def apply_filters(
    results: list,
    filters: RetrievalFilters,
) -> list:
    """Filter retrieval results based on metadata criteria.

    Works with both HybridResult and RetrievedChunk objects.

    Args:
        results: List of retrieval results.
        filters: Filter specification.

    Returns:
        Filtered list of results.
    """
    if not filters:
        return results

    filtered = []
    for result in results:
        if _matches_filters(result, filters):
            filtered.append(result)

    logger.debug(
        "Filters applied",
        extra={"extra_data": {
            "input_count": len(results),
            "output_count": len(filtered),
            "filters": _filters_to_dict(filters),
        }},
    )

    return filtered


def _matches_filters(result, filters: RetrievalFilters) -> bool:
    """Check if a result matches all non-None filter criteria."""
    if filters.paper_ids and getattr(result, "paper_id", None) not in filters.paper_ids:
        return False

    if filters.sections:
        sec = getattr(result, "section", None)
        if sec is None or sec not in filters.sections:
            # Also check subsection
            subsec = getattr(result, "subsection", None)
            if subsec is None or subsec not in filters.sections:
                return False

    if filters.year_min is not None or filters.year_max is not None:
        year = None
        meta = getattr(result, "metadata", {})
        if isinstance(meta, dict):
            year = meta.get("year")
        if year is None:
            return False
        if filters.year_min is not None and year < filters.year_min:
            return False
        if filters.year_max is not None and year > filters.year_max:
            return False

    if filters.sources:
        source = None
        meta = getattr(result, "metadata", {})
        if isinstance(meta, dict):
            source = meta.get("source")
        if source is None or source not in filters.sources:
            return False

    return True


def _filters_to_dict(filters: RetrievalFilters) -> dict:
    """Convert filters to a dict for logging."""
    return {
        "paper_ids": filters.paper_ids,
        "sources": filters.sources,
        "year_min": filters.year_min,
        "year_max": filters.year_max,
        "sections": filters.sections,
        "authors": filters.authors,
    }


def build_qdrant_filter(filters: RetrievalFilters) -> Optional[dict]:
    """Build a Qdrant-compatible filter dict from RetrievalFilters."""
    conditions: dict = {}

    if filters.paper_ids and len(filters.paper_ids) == 1:
        conditions["paper_id"] = filters.paper_ids[0]
    elif filters.paper_ids:
        conditions["paper_ids"] = filters.paper_ids

    if filters.sources:
        conditions["source"] = filters.sources if len(filters.sources) > 1 else filters.sources[0]

    return conditions if conditions else None
