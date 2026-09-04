"""Context builder — constructs structured LLM context from retrieval results.

Takes reranked chunks and formats them into a structured context
that clearly separates sources, includes section/page references,
removes duplicates, and respects context size limits.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings
from app.retrieval.hybrid import HybridResult

logger = logging.getLogger("raglens.generation.context")


@dataclass
class SourceReference:
    """A source reference in the generated context."""

    citation_id: int
    chunk_id: str
    paper_id: str
    title: str
    section: Optional[str]
    subsection: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    score: float


@dataclass
class BuiltContext:
    """The built context ready for LLM generation."""

    context_text: str
    sources: list[SourceReference]
    total_tokens: int
    truncated: bool


class ContextBuilder:
    """Builds structured context from retrieval results for LLM generation.

    Args:
        max_tokens: Maximum context size (default 4000 tokens, fitting CPU LLMs).
        chunk_separator: String between source blocks in context.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        chunk_separator: str = "\n\n",
    ) -> None:
        self._max_tokens = max_tokens
        self._chunk_separator = chunk_separator

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ContextBuilder":
        """Create a context builder from application settings."""
        if settings is None:
            settings = Settings()
        max_tokens = getattr(settings, "llm_max_tokens", 4000)
        # Reserve space for the prompt and response
        context_tokens = max(500, max_tokens - 1024)
        return cls(max_tokens=context_tokens)

    def build(
        self,
        query: str,
        results: list[HybridResult],
    ) -> BuiltContext:
        """Build a structured context from retrieval results.

        Args:
            query: The original user query.
            results: List of HybridResult objects (reranked).

        Returns:
            A BuiltContext with formatted context text and source references.
        """
        sources: list[SourceReference] = []
        context_parts: list[str] = []
        total_tokens = 0
        truncated = False

        # Deduplicate by chunk_id
        seen_chunks: set[str] = set()
        unique_results: list[HybridResult] = []
        for r in results:
            if r.chunk_id not in seen_chunks:
                seen_chunks.add(r.chunk_id)
                unique_results.append(r)

        for i, result in enumerate(unique_results, 1):
            source = SourceReference(
                citation_id=i,
                chunk_id=result.chunk_id,
                paper_id=result.paper_id,
                title=result.title or "Untitled Paper",
                section=result.section,
                subsection=result.subsection,
                page_start=result.page_start,
                page_end=result.page_end,
                text=result.text[:500],  # Truncate very long chunks
                score=result.score,
            )
            sources.append(source)

            # Format source block
            section_ref = source.section or "N/A"
            page_ref = f"Page {source.page_start}" if source.page_start is not None else ""

            block = (
                f"SOURCE [{i}]\nPaper: {source.title}\n"
                f"Section: {section_ref} {page_ref}\n"
                f"Evidence:\n{source.text}"
            )

            block_tokens = self._estimate_tokens(block)
            if total_tokens + block_tokens > self._max_tokens:
                # Partially include this block if space allows
                remaining = self._max_tokens - total_tokens
                if remaining > 100:
                    truncated_block = self._truncate_text(source.text, remaining)
                    block = (
                        f"SOURCE [{i}]\nPaper: {source.title}\n"
                        f"Section: {section_ref} {page_ref}\n"
                        f"Evidence:\n{truncated_block}"
                    )
                    context_parts.append(block)
                    total_tokens += self._estimate_tokens(block)
                truncated = True
                break

            context_parts.append(block)
            total_tokens += block_tokens

        # Prepend the query instruction
        instruction = (
            f"Context for answering the following question:\n"
            f"Question: {query}\n\n"
            f"Use only the evidence below. Do not hallucinate facts. "
            f"Cite sources using [N] format.\n\n"
        )

        context_text = instruction + self._chunk_separator.join(context_parts)
        total_tokens += self._estimate_tokens(instruction)

        logger.info(
            "Context built",
            extra={"extra_data": {
                "sources": len(sources),
                "total_tokens": total_tokens,
                "truncated": truncated,
            }},
        )

        return BuiltContext(
            context_text=context_text,
            sources=sources,
            total_tokens=total_tokens,
            truncated=truncated,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return max(1, len(text) // 4)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 50] + " [truncated]"


# Re-export SourceReference for citation engine use
__all__ = ["ContextBuilder", "BuiltContext", "SourceReference"]
