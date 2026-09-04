"""Literature review pipeline — generates synthesized literature reviews."""

from __future__ import annotations

import logging
from typing import Optional

from app.generation.context_builder import ContextBuilder
from app.generation.citation import CitationEngine, CitationValidator
from app.generation.llm import LLMProvider, get_llm_provider
from app.generation.prompts import LITERATURE_REVIEW_PROMPT_TEMPLATE
from app.core.config import Settings, get_settings
from app.services.search_service import SearchService

logger = logging.getLogger("raglens.pipeline.literature")


class LiteratureReviewPipeline:
    """Pipeline for generating literature reviews.

    Retrieves relevant papers, identifies themes, and generates
    a structured synthesis with citations.

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

    def review(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> dict:
        """Generate a literature review on a topic.

        Args:
            query: The review topic/question.
            top_k: Number of papers to retrieve (defaults to setting).

        Returns:
            Dict with review text, citations, evidence, and retrieval info.
        """
        if top_k is None:
            top_k = self._settings.hybrid_top_k

        # Retrieve broadly across the corpus
        results = self._search_service.search(
            query=query,
            top_k=top_k,
            filters=None,
            rerank=True,
        )

        if not results:
            return {
                "answer": (
                    "I couldn't find sufficient evidence in the indexed papers "
                    "to conduct a literature review on this topic."
                ),
                "citations": [],
                "evidence": [],
                "retrieval": {"query": query, "results": 0},
                "insufficient_evidence": True,
            }

        # Build context from all retrieved papers
        built_context = self._context_builder.build(query, results)

        # Generate citations
        citation_engine = CitationEngine()
        citation_map = citation_engine.generate_citations(built_context.sources)

        prompt = LITERATURE_REVIEW_PROMPT_TEMPLATE.format(
            query=query,
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

        # Count unique papers referenced
        paper_ids = set()
        for r in results:
            paper_ids.add(r.paper_id)

        return {
            "answer": sanitized_text,
            "citations": citations,
            "evidence": evidence,
            "retrieval": {
                "query": query,
                "total_results": len(results),
                "unique_papers": len(paper_ids),
                "context_tokens": built_context.total_tokens,
            },
            "insufficient_evidence": False,
        }
