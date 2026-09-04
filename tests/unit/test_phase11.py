"""Tests for query classification (Phase 11)."""

from __future__ import annotations

from app.retrieval.query_classifier import QueryClassifier, QueryIntent


class TestQueryClassifier:
    """Test the query classifier."""

    def setup_method(self) -> None:
        self.classifier = QueryClassifier()

    def test_classify_summary_query(self) -> None:
        result = self.classifier.classify("Summarize this paper")
        assert result == QueryIntent.PAPER_SUMMARY

    def test_classify_tl_dr(self) -> None:
        result = self.classifier.classify("Give me a TLDR of the methodology")
        assert result == QueryIntent.PAPER_SUMMARY

    def test_classify_comparison_query(self) -> None:
        result = self.classifier.classify("Compare Self-RAG and CRAG")
        assert result == QueryIntent.PAPER_COMPARISON

    def test_classify_versus_query(self) -> None:
        result = self.classifier.classify("What are the differences between dense and sparse retrieval?")
        assert result == QueryIntent.PAPER_COMPARISON

    def test_classify_contrast_query(self) -> None:
        result = self.classifier.classify("Contrast the approaches of these two papers")
        assert result == QueryIntent.PAPER_COMPARISON

    def test_classify_literature_review_query(self) -> None:
        result = self.classifier.classify("Literature review of RAG evaluation techniques")
        assert result == QueryIntent.LITERATURE_REVIEW

    def test_classify_overview_query(self) -> None:
        result = self.classifier.classify("What are the major approaches to improving retrieval quality?")
        assert result == QueryIntent.LITERATURE_REVIEW

    def test_classify_evidence_query(self) -> None:
        result = self.classifier.classify("Find evidence supporting the claim that hybrid retrieval improves quality")
        assert result == QueryIntent.EVIDENCE_SEARCH

    def test_classify_evidence_query_2(self) -> None:
        result = self.classifier.classify("What evidence shows that reranking helps?")
        assert result == QueryIntent.EVIDENCE_SEARCH

    def test_classify_paper_specific_query(self) -> None:
        result = self.classifier.classify("What datasets did this paper use?")
        assert result == QueryIntent.PAPER_SPECIFIC_QUERY

    def test_classify_paper_specific_query_2(self) -> None:
        result = self.classifier.classify("What are the limitations of this study?")
        assert result == QueryIntent.PAPER_SPECIFIC_QUERY

    def test_classify_general_qa_query(self) -> None:
        result = self.classifier.classify("What is hybrid retrieval?")
        assert result == QueryIntent.QUESTION_ANSWERING

    def test_classify_bm25_exact_term(self) -> None:
        result = self.classifier.classify("What is BM25?")
        assert result == QueryIntent.QUESTION_ANSWERING

    def test_classify_self_rag(self) -> None:
        result = self.classifier.classify("How does Self-RAG work?")
        assert result == QueryIntent.QUESTION_ANSWERING

    def test_classify_datasets_query(self) -> None:
        result = self.classifier.classify("What datasets are commonly used to evaluate RAG systems?")
        assert result == QueryIntent.LITERATURE_REVIEW

    def test_classify_limitations_query(self) -> None:
        result = self.classifier.classify("What are the limitations identified by these papers?")
        assert result == QueryIntent.LITERATURE_REVIEW

    def test_classify_reranking_query(self) -> None:
        result = self.classifier.classify("Which papers discuss reranking?")
        assert result == QueryIntent.EVIDENCE_SEARCH

    def test_classify_with_confidence(self) -> None:
        intent, confidence = self.classifier.classify_with_confidence("Summarize this paper")
        assert intent == QueryIntent.PAPER_SUMMARY
        assert confidence > 0.5

    def test_classify_qa_with_confidence(self) -> None:
        intent, confidence = self.classifier.classify_with_confidence("What is dense retrieval?")
        assert intent == QueryIntent.QUESTION_ANSWERING
        assert confidence > 0.0
