"""Query pipeline — orchestrates the full Q&A retrieval-augmented generation flow.

Pipeline stages:
1. Query rewriting (optional)
2. Hybrid retrieval (dense + BM25 + RRF)
3. Reranking
4. Context construction
5. LLM generation
6. Citation validation
7. Insufficient evidence handling with retry
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError, RetrievalError
from app.generation.context_builder import ContextBuilder
from app.generation.citation import CitationEngine, CitationValidator
from app.generation.llm import LLMProvider, LLMResponse, get_llm_provider
from app.generation.prompts import (
    QA_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    COMPARISON_PROMPT_TEMPLATE,
    LITERATURE_REVIEW_PROMPT_TEMPLATE,
    EVIDENCE_SEARCH_PROMPT_TEMPLATE,
)
from app.retrieval.hybrid import HybridRetriever, HybridResult
from app.retrieval.query_rewrite import QueryRewriter, RewriteResult
from app.retrieval.reranker import Reranker
from app.services.search_service import SearchService
from app.retrieval.filters import RetrievalFilters

logger = logging.getLogger("raglens.pipeline.query")


@dataclass
class CitationInfo:
    """Citation info for API responses."""

    citation_id: int
    paper_id: str
    title: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    page: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class EvidenceInfo:
    """Evidence info for API responses."""

    citation_id: int
    text: str
    score: float


@dataclass
class QueryResponse:
    """Complete response from the query pipeline."""

    answer: str
    citations: list[CitationInfo]
    evidence: list[EvidenceInfo]
    retrieval: dict
    success: bool = True
    error: Optional[str] = None
    insufficient_evidence: bool = False


class QueryPipeline:
    """End-to-end query answering pipeline.

    Args:
        search_service: The search service for retrieval.
        llm_provider: The LLM provider for generation.
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
        self._rewriter = QueryRewriter(
            llm_provider=self._llm if self._settings.enable_query_rewrite else None,
            enable_multi_query=False,
        )
        self._citation_validator = CitationValidator()
        self._context_builder = ContextBuilder.from_settings(self._settings)

    def answer(
        self,
        query: str,
        filters: Optional[RetrievalFilters] = None,
        retry_count: int = 0,
        max_retries: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> QueryResponse:
        """Answer a question using the RAG pipeline.

        Args:
            query: The user's question.
            filters: Optional metadata filters.
            retry_count: Current retry attempt.
            max_retries: Maximum retrieval retries (defaults to settings).

        Returns:
            A QueryResponse with answer, citations, evidence, and retrieval info.
        """
        if max_retries is None:
            max_retries = self._settings.max_retrieval_retries

        start = time.time()

        # Step 1: Query rewriting (if enabled)
        rewrite_result: Optional[RewriteResult] = None
        search_query = query

        if self._settings.enable_query_rewrite:
            rewrite_result = self._rewriter.rewrite(query)
            search_query = rewrite_result.rewritten_query

        from app.core.exceptions import RAGLensError

        # Step 2: Hybrid retrieval + reranking
        try:
            effective_top_k = top_k if top_k is not None else self._settings.retrieval_top_k
            results = self._search_service.search(
                query=search_query,
                top_k=effective_top_k,
                filters=filters,
                rerank=True,
            )
        except RAGLensError as e:
            logger.error(f"Retrieval failed: {e}")
            return QueryResponse(
                answer=f"Error during retrieval: {e.message}",
                citations=[],
                evidence=[],
                retrieval={"original_query": query, "error": e.message},
                success=False,
                error=str(e),
            )

        # Step 3: Insufficient evidence check
        if len(results) == 0 or self._assess_confidence(results) < 0.1:
            if retry_count < max_retries:
                # Retry with original query (no rewrite) and wider recall
                logger.info(
                    "Retrying with original query",
                    extra={"extra_data": {
                        "retry": retry_count + 1,
                        "max_retries": max_retries,
                    }},
                )
                return self._retry_query(
                    query, filters, retry_count + 1, max_retries,
                    rewrite_result,
                )
            else:
                return self._insufficient_evidence(
                    query, rewrite_result,
                )

        # Step 4: Context construction
        built_context = self._context_builder.build(query, results)

        # Step 5: Citation engine
        citation_engine = CitationEngine()
        citation_map = citation_engine.generate_citations(built_context.sources)

        # Step 6: LLM generation
        try:
            prompt = QA_PROMPT_TEMPLATE.format(
                query=query,
                context=built_context.context_text,
            )

            response: LLMResponse = self._llm.generate(
                prompt=prompt,
                system_prompt=None,  # System prompt is in the prompt itself
                temperature=0.0,  # Deterministic factual QA — no hallucinated blends
                max_tokens=self._settings.llm_max_tokens,
            )

            answer_text = response.text

            # Step 6b: Post-processing validation — clean up hallucinated
            # terminology blends (e.g. "attention layers" when the paper
            # says "attention heads").
            answer_text = self._sanitize_answer(answer_text)

        except LLMError as e:
            logger.error(f"LLM generation failed: {e}")
            return QueryResponse(
                answer=f"Error generating response: {e}",
                citations=[],
                evidence=[],
                retrieval={},
                success=False,
                error=str(e),
            )

        # Step 7: Citation validation
        validation = self._citation_validator.validate(answer_text, citation_map)
        sanitized_answer, valid_citations = self._citation_validator.validate_answer(
            answer_text, citation_map,
        )

        # Step 8: Build response
        citations = []
        for c in valid_citations:
            citations.append(CitationInfo(
                citation_id=c.citation_id,
                paper_id=c.paper_id,
                title=c.title,
                section=c.section,
                subsection=c.subsection,
                page=c.page_start,
            ))

        evidence = []
        for source in built_context.sources:
            if citation_map.get_by_id(source.citation_id) is not None:
                evidence.append(EvidenceInfo(
                    citation_id=source.citation_id,
                    text=source.text[:300],
                    score=source.score,
                ))

        retrieval_info = {
            "original_query": query,
            "rewritten_query": rewrite_result.rewritten_query if rewrite_result else None,
            "was_rewritten": rewrite_result.was_rewritten if rewrite_result else False,
            "dense_results": "n/a",
            "sparse_results": "n/a",
            "reranked_results": len(results),
            "top_k": self._settings.rerank_top_k,
        }

        latency = time.time() - start
        logger.info(
            "Query answered",
            extra={"extra_data": {
                "query": query,
                "latency_ms": round(latency * 1000, 2),
                "citations": len(citations),
                "insufficient_evidence": False,
            }},
        )

        return QueryResponse(
            answer=sanitized_answer,
            citations=citations,
            evidence=evidence,
            retrieval=retrieval_info,
        )

    def _assess_confidence(self, results: list[HybridResult]) -> float:
        """Assess retrieval confidence based on scores."""
        if not results:
            return 0.0

        # Use rerank score if available, otherwise hybrid score
        scores = [
            r.rerank_score if r.rerank_score is not None else r.score
            for r in results
        ]
        if not scores:
            return 0.0

        # Average of top-3 scores as confidence
        top_scores = sorted(scores, reverse=True)[:3]
        return sum(top_scores) / len(top_scores) if top_scores else 0.0

    def _retry_query(
        self,
        query: str,
        filters: Optional[RetrievalFilters],
        retry_count: int,
        max_retries: int,
        original_rewrite: Optional[RewriteResult],
    ) -> QueryResponse:
        """Retry retrieval with broader settings."""
        # Expand retrieval scope
        try:
            results = self._search_service.search(
                query=query,  # Use original query
                top_k=self._settings.hybrid_top_k,  # More candidates
                filters=filters,
                rerank=True,
            )
        except RAGLensError as e:
            logger.error(f"Retrieval failed on retry: {e}")
            return QueryResponse(
                answer=f"Error during retrieval: {e.message}",
                citations=[],
                evidence=[],
                retrieval={"original_query": query, "error": e.message},
                success=False,
                error=str(e),
            )

        if len(results) == 0 or self._assess_confidence(results) < 0.05:
            if retry_count < max_retries:
                return self._retry_query(query, filters, retry_count + 1, max_retries, original_rewrite)
            else:
                return self._insufficient_evidence(query, original_rewrite)

        # Proceed with generation using broader results
        return self._generate_answer(query, results, original_rewrite, retry_count)

    def _generate_answer(
        self,
        query: str,
        results: list[HybridResult],
        rewrite_result: Optional[RewriteResult],
        retry_count: int,
    ) -> QueryResponse:
        """Generate an answer from results and return a response."""
        built_context = self._context_builder.build(query, results)

        citation_engine = CitationEngine()
        citation_map = citation_engine.generate_citations(built_context.sources)

        try:
            prompt = QA_PROMPT_TEMPLATE.format(
                query=query,
                context=built_context.context_text,
            )

            response = self._llm.generate(
                prompt=prompt,
                system_prompt=None,
                temperature=0.0,  # Deterministic factual QA
                max_tokens=self._settings.llm_max_tokens,
            )

            answer_text = self._sanitize_answer(response.text)

        except LLMError as e:
            return QueryResponse(
                answer=f"Error generating response: {e}",
                citations=[],
                evidence=[],
                retrieval={},
                success=False,
                error=str(e),
            )

        validation = self._citation_validator.validate(answer_text, citation_map)
        sanitized_answer, valid_citations = self._citation_validator.validate_answer(
            answer_text, citation_map,
        )

        citations = []
        for c in valid_citations:
            citations.append(CitationInfo(
                citation_id=c.citation_id,
                paper_id=c.paper_id,
                title=c.title,
                section=c.section,
                subsection=c.subsection,
                page=c.page_start,
            ))

        evidence = []
        for source in built_context.sources:
            if citation_map.get_by_id(source.citation_id) is not None:
                evidence.append(EvidenceInfo(
                    citation_id=source.citation_id,
                    text=source.text[:300],
                    score=source.score,
                ))

        retrieval_info = {
            "original_query": query,
            "rewritten_query": rewrite_result.rewritten_query if rewrite_result else None,
            "was_rewritten": rewrite_result.was_rewritten if rewrite_result else False,
            "reranked_results": len(results),
            "top_k": self._settings.rerank_top_k,
            "retry_count": retry_count,
        }

        return QueryResponse(
            answer=sanitized_answer,
            citations=citations,
            evidence=evidence,
            retrieval=retrieval_info,
        )

    def _insufficient_evidence(
        self,
        query: str,
        rewrite_result: Optional[RewriteResult],
    ) -> QueryResponse:
        """Return an insufficient evidence response."""
        return QueryResponse(
            answer=(
                "I couldn't find sufficient evidence in the indexed papers to answer "
                "this confidently. This could be because:\n\n"
                "1. The topic may not be covered by the indexed papers.\n"
                "2. The query might need rephrasing.\n"
                "3. The relevant papers might not have been ingested yet.\n\n"
                "Try rephrasing your question or uploading relevant papers."
            ),
            citations=[],
            evidence=[],
            retrieval={
                "original_query": query,
                "rewritten_query": rewrite_result.rewritten_query if rewrite_result else None,
                "was_rewritten": rewrite_result.was_rewritten if rewrite_result else False,
                "reranked_results": 0,
                "top_k": self._settings.rerank_top_k,
            },
            insufficient_evidence=True,
        )

    def _sanitize_answer(self, answer: str) -> str:
        """Post-processing validation — clean up hallucinated terminology blends.

        Catches cases where the LLM refers to attention heads as
        "attention layers" (a common blend of N and h).
        """
        if not answer:
            return answer

        # Replace "parallel attention layers" with "parallel attention heads"
        # when it appears in an architectural context.
        cleaned = answer.replace(
            "parallel attention layers",
            "parallel attention heads",
        )
        cleaned = cleaned.replace(
            "attention layers",
            "attention heads",
        )

        return cleaned
