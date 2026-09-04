"""Evidence search pipeline — finds supporting evidence for claims."""

from __future__ import annotations

import logging
from typing import Optional

from app.generation.context_builder import ContextBuilder
from app.generation.citation import CitationEngine, CitationValidator
from app.generation.llm import LLMProvider, get_llm_provider
from app.generation.prompts import EVIDENCE_SEARCH_PROMPT_TEMPLATE
from app.core.config import Settings, get_settings
from app.retrieval.filters import RetrievalFilters
from app.services.search_service import SearchService

logger = logging.getLogger("raglens.pipeline.evidence")


class EvidencePipeline:
    """Pipeline for searching evidence supporting or refuting claims.

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

    def search_evidence(
        self,
        claim: str,
        top_k: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None,
    ) -> dict:
        """Search for evidence supporting or refuting a claim.

        Args:
            claim: The claim to investigate (e.g., "hybrid retrieval improves quality").
            top_k: Number of results to retrieve.
            filters: Optional metadata filters.

        Returns:
            Dict with evidence analysis, citations, and sources.
        """
        if top_k is None:
            top_k = self._settings.hybrid_top_k

        # Retrieve evidence
        results = self._search_service.search(
            query=claim,
            top_k=top_k,
            filters=filters,
            rerank=True,
        )

        if not results:
            return {
                "answer": (
                    f"I couldn't find evidence in the indexed papers related to: "
                    f"'{claim}'"
                ),
                "supporting": [],
                "contradicting": [],
                "confidence": "low",
                "citations": [],
                "evidence": [],
            }

        # Build context
        built_context = self._context_builder.build(claim, results)

        # Generate citations
        citation_engine = CitationEngine()
        citation_map = citation_engine.generate_citations(built_context.sources)

        prompt = EVIDENCE_SEARCH_PROMPT_TEMPLATE.format(
            claim=claim,
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
            "answer": sanitized_text,
            "supporting": [],
            "contradicting": [],
            "confidence": "medium",
            "citations": citations,
            "evidence": evidence,
            "retrieval": {
                "claim": claim,
                "results": len(results),
                "context_tokens": built_context.total_tokens,
            },
        }
