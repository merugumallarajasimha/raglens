"""Query classification — categorizes user queries into intent types.

Uses lightweight keyword-based classification to route queries to
the appropriate pipeline (Q&A, summary, comparison, literature review, etc.).
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger("raglens.retrieval.query_classifier")


class QueryIntent(str, Enum):
    """Types of queries the system can handle."""

    QUESTION_ANSWERING = "question_answering"
    PAPER_SUMMARY = "paper_summary"
    PAPER_COMPARISON = "paper_comparison"
    LITERATURE_REVIEW = "literature_review"
    EVIDENCE_SEARCH = "evidence_search"
    PAPER_SPECIFIC_QUERY = "paper_specific_query"


class QueryClassifier:
    """Lightweight query classifier using keyword-based heuristics.

    Uses simple pattern matching to classify queries into intent types.
    This avoids the overhead of an LLM for classification.
    """

    # Keyword patterns for each intent type
    _PATTERNS = {
        QueryIntent.PAPER_SUMMARY: re.compile(
            r"\b(summariz|summarise|summarise this|summary|tldr|give me.*overview)"
            r"|\bsummar.*\b(this|it|the paper|this paper|this article)\b"
            r"|\bwhat are the (main |key |major )?(findings|contributions|points)\b",
            re.IGNORECASE,
        ),
        QueryIntent.PAPER_COMPARISON: re.compile(
            r"\b(compare|comparing|comparison|versus|vs|differenc|contrast|trade-off|tradeoff)",
            re.IGNORECASE,
        ),
        QueryIntent.LITERATURE_REVIEW: re.compile(
            r"\b(literature review|survey|state of the art|state-of-the-art|overview of)"
            r"|\bwhat are the (approaches|methods|techniques|strategies)\b"
            r"|\b(major|key|main) (approaches|methods|techniques|trends)\b"
            r"|\bwhat (datasets|dataset|metrics|methods|techniques|approaches)"
            r"|\bcommonly used to (evaluate|train|benchmark)\b"
            r"|\blimitations\b"
            r"|\bwhat (are|is) the (main|key|major) (limitations|challenges|problems|issues|gaps)\b"
            r"|\bgaps\b"
            r"\bwhat (are|is) (the )?(current|recent|major|key) (research|open) (directions|challenges|gaps)\b",
            re.IGNORECASE,
        ),
        QueryIntent.EVIDENCE_SEARCH: re.compile(
            r"\b(evidence|support|backed|cite|show that|proven|demonstrated)"
            r"|\bwhich papers( discuss| talk about| mention| cover| address)\b"
            r"|\bpapers discuss\b"
            r"|\bwhat (papers|studies|research)\b",
            re.IGNORECASE,
        ),
        QueryIntent.PAPER_SPECIFIC_QUERY: re.compile(
            r"\b(this paper|the paper|this study|the study|this research)\b"
            r"|\b(paper_id|paper id)\b",
            re.IGNORECASE,
        ),
    }

    def classify(self, query: str) -> QueryIntent:
        """Classify a query into an intent type.

        Args:
            query: The user query to classify.

        Returns:
            A QueryIntent enum value.
        """
        query_lower = query.lower().strip()

        # Check each intent pattern
        # Order matters: more specific patterns first
        for intent, pattern in [
            (QueryIntent.PAPER_SUMMARY, self._PATTERNS[QueryIntent.PAPER_SUMMARY]),
            (QueryIntent.PAPER_COMPARISON, self._PATTERNS[QueryIntent.PAPER_COMPARISON]),
            (QueryIntent.EVIDENCE_SEARCH, self._PATTERNS[QueryIntent.EVIDENCE_SEARCH]),
            (QueryIntent.PAPER_SPECIFIC_QUERY, self._PATTERNS[QueryIntent.PAPER_SPECIFIC_QUERY]),
            (QueryIntent.LITERATURE_REVIEW, self._PATTERNS[QueryIntent.LITERATURE_REVIEW]),
        ]:
            if pattern.search(query_lower):
                logger.debug(
                    f"Query classified as {intent.value}",
                    extra={"extra_data": {"query": query, "intent": intent.value}},
                )
                return intent

        # Default: question answering
        return QueryIntent.QUESTION_ANSWERING

    def classify_with_confidence(self, query: str) -> tuple[QueryIntent, float]:
        """Classify a query and return a confidence score.

        Args:
            query: The user query to classify.

        Returns:
            Tuple of (QueryIntent, confidence_score) where confidence is in [0, 1].
        """
        intent = self.classify(query)

        # Compute confidence based on pattern match strength
        query_lower = query.lower()
        pattern = self._PATTERNS.get(intent)

        if intent == QueryIntent.QUESTION_ANSWERING:
            return intent, 0.7  # Default confidence

        if pattern and pattern.search(query_lower):
            return intent, 0.85
        return intent, 0.5  # Low confidence fallback
