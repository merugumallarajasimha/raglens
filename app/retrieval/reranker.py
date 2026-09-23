"""Reranking using cross-encoder models.

Applies a cross-encoder to re-score retrieved candidates based on
query-candidate interaction, producing more precise relevance scores.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.core.config import Settings
from app.core.exceptions import RerankError
from app.retrieval.hybrid import HybridResult

logger = logging.getLogger("raglens.retrieval.reranker")


class Reranker:
    """Re-ranks retrieved chunks using a cross-encoder model.

    Args:
        model_name: Name of the cross-encoder model.
        provider: Provider type ("sentence_transformers" or "mock").
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        provider: str = "sentence_transformers",
    ) -> None:
        self._model_name = model_name
        self._provider = provider
        self._model = None

        if provider == "sentence_transformers":
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker model: {model_name}")
                self._model = CrossEncoder(model_name)
                logger.info("Reranker model loaded")
            except Exception as e:
                logger.warning(
                    f"Failed to load reranker model {model_name}: {e}. "
                    "Falling back to mock reranker."
                )
                self._model = None
        elif provider == "mock":
            self._model = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "Reranker":
        """Create a reranker from application settings."""
        return cls(
            model_name=settings.reranker_model,
            provider=settings.reranker_provider,
        )

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_k: int = 10,
    ) -> list[HybridResult]:
        """Re-rank candidates using the cross-encoder.

        Args:
            query: The original search query.
            candidates: List of HybridResult objects from hybrid retrieval.
            top_k: Number of results to return after reranking.

        Returns:
            Re-ranked list of HybridResult objects (sorted by rerank score).

        Raises:
            RerankError: If reranking fails.
        """
        if not candidates:
            return []

        start = time.time()

        try:
            if self._model is not None:
                # Use cross-encoder for re-scoring
                pairs = [(query, c.text) for c in candidates]
                scores = self._model.predict(pairs, show_progress_bar=False)

                # Update scores with reranker scores
                for candidate, score in zip(candidates, scores):
                    candidate.rerank_score = float(score)

                candidates.sort(key=lambda c: c.rerank_score, reverse=True)
            else:
                # Mock reranker: use a simple text overlap heuristic
                # This provides deterministic results for testing
                self._mock_rerank(query, candidates)
                candidates.sort(key=lambda c: c.rerank_score, reverse=True)

            results = candidates[:top_k]

            latency = time.time() - start
            logger.info(
                "Reranking complete",
                extra={"extra_data": {
                    "query": query,
                    "candidates": len(candidates),
                    "top_k": top_k,
                    "latency_ms": round(latency * 1000, 2),
                }},
            )

            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise RerankError(f"Reranking failed: {e}") from e

    def _mock_rerank(self, query: str, candidates: list[HybridResult]) -> None:
        """Simple text overlap-based reranking for testing.

        Uses token overlap between query and candidate text as a proxy for
        relevance. This is not as accurate as a cross-encoder but provides
        a deterministic baseline.
        """
        query_tokens = set(query.lower().split())
        for candidate in candidates:
            cand_tokens = set(candidate.text.lower().split())
            overlap = len(query_tokens & cand_tokens)
            total = len(query_tokens | cand_tokens)
            candidate.rerank_score = overlap / total if total > 0 else 0.0
