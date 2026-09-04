"""Paper summarization pipeline — generates structured paper summaries."""

from __future__ import annotations

import logging
from typing import Optional

from app.generation.context_builder import ContextBuilder
from app.generation.citation import CitationEngine, CitationValidator
from app.generation.llm import LLMProvider, get_llm_provider
from app.generation.prompts import SUMMARY_PROMPT_TEMPLATE
from app.core.config import Settings, get_settings
from app.retrieval.filters import RetrievalFilters
from app.services.search_service import SearchService

logger = logging.getLogger("raglens.pipeline.summary")


class SummaryPipeline:
    """Pipeline for generating structured paper summaries.

    Args:
        search_service: Search service for retrieval.
        llm_provider: LLM provider for generation.
        settings: Application settings.
    """

    def __init__(
        self,
        search_service: SearchService,
        llm_provider: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._search_service = search_service
        self._llm = llm_provider or get_llm_provider(self._settings)
        self._context_builder = ContextBuilder.from_settings(self._settings)
        self._citation_validator = CitationValidator()

    def summarize(
        self,
        paper_id: str,
        query: str = "Summarize this paper",
    ) -> dict:
        """Generate a structured summary for a single paper.

        Retrieves relevant sections from the paper and generates a
        structured summary with citations to source evidence.

        Args:
            paper_id: The ID of the paper to summarize.
            query: Optional additional context for the summary focus.

        Returns:
            Dict with summary, citations, evidence, and retrieval info.
        """
        # Retrieve content from the specific paper
        filters = RetrievalFilters(paper_ids=[paper_id])
        results = self._search_service.search(
            query=query,
            top_k=self._settings.hybrid_top_k,
            filters=filters,
            rerank=True,
        )

        if not results:
            return {
                "summary": "No content found for this paper.",
                "citations": [],
                "evidence": [],
                "retrieval": {"paper_id": paper_id, "results": 0},
                "insufficient_evidence": True,
            }

        # Get paper title
        title = results[0].title if results else "Untitled Paper"

        # Build context
        built_context = self._context_builder.build(query, results)

        # Generate citations
        citation_engine = CitationEngine()
        citation_map = citation_engine.generate_citations(built_context.sources)

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            title=title,
            context=built_context.context_text,
        )

        response = self._llm.generate(
            prompt=prompt,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )

        # Validate citations
        sanitized_text, valid_citations = self._citation_validator.validate_answer(
            response.text, citation_map,
        )

        citations = []
        for c in valid_citations:
            citations.append({
                "citation_id": c.citation_id,
                "paper_id": c.paper_id,
                "title": c.title,
                "section": c.section,
                "page": c.page_start,
            })

        evidence = []
        for source in built_context.sources:
            if self._citation_validator.validate_citation_reference(source.citation_id):
                evidence.append({
                    "citation_id": source.citation_id,
                    "text": source.text[:300],
                    "score": source.score,
                })

        return {
            "summary": sanitized_text,
            "citations": citations,
            "evidence": evidence,
            "retrieval": {
                "paper_id": paper_id,
                "title": title,
                "results": len(results),
                "context_tokens": built_context.total_tokens,
            },
            "insufficient_evidence": False,
        }
