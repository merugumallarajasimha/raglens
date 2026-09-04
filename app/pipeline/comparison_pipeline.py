"""Paper comparison pipeline — compares multiple papers using retrieved evidence."""

from __future__ import annotations

import logging
from typing import Optional

from app.generation.context_builder import ContextBuilder
from app.generation.citation import CitationEngine, CitationValidator
from app.generation.llm import LLMProvider, get_llm_provider
from app.generation.prompts import COMPARISON_PROMPT_TEMPLATE
from app.core.config import Settings, get_settings
from app.retrieval.hybrid import HybridResult
from app.services.search_service import SearchService
from app.retrieval.filters import RetrievalFilters

logger = logging.getLogger("raglens.pipeline.comparison")


class ComparisonPipeline:
    """Pipeline for comparing multiple papers.

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

    def compare(
        self,
        paper_ids: list[str],
        query: str = "Compare these papers",
    ) -> dict:
        """Compare multiple papers on a topic.

        Args:
            paper_ids: List of paper IDs to compare.
            query: The comparison question.

        Returns:
            Dict with comparison answer, citations, and evidence.
        """
        if len(paper_ids) < 2:
            raise ValueError("At least 2 papers are required for comparison")

        if len(paper_ids) > 4:
            raise ValueError("At most 4 papers can be compared at once")

        # Retrieve relevant content from each paper
        all_results: list[HybridResult] = []
        for paper_id in paper_ids:
            filters = RetrievalFilters(paper_ids=[paper_id])
            results = self._search_service.search(
                query=query,
                top_k=self._settings.rerank_top_k,
                filters=filters,
                rerank=True,
            )
            all_results.extend(results)

        if not all_results:
            return {
                "answer": "I couldn't find sufficient evidence about these papers to compare them.",
                "citations": [],
                "evidence": [],
                "retrieval": {"paper_ids": paper_ids, "results_per_paper": 0},
                "insufficient_evidence": True,
            }

        # Build context
        built_context = self._context_builder.build(query, all_results)

        # Generate citations
        citation_engine = CitationEngine()
        citation_map = citation_engine.generate_citations(built_context.sources)

        # Get paper titles for the prompt
        paper_titles = {}
        for r in all_results:
            if r.paper_id not in paper_titles:
                paper_titles[r.paper_id] = r.title

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            paper_a_title=paper_titles.get(paper_ids[0], paper_ids[0]),
            paper_b_title=paper_titles.get(paper_ids[1], paper_ids[1]),
            context=built_context.context_text,
        )

        response = self._llm.generate(
            prompt=prompt,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )

        # Validate citations
        sanitized_answer, valid_citations = self._citation_validator.validate_answer(
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
            "answer": sanitized_answer,
            "citations": citations,
            "evidence": evidence,
            "retrieval": {
                "paper_ids": paper_ids,
                "total_results": len(all_results),
                "context_tokens": built_context.total_tokens,
            },
            "insufficient_evidence": False,
        }
