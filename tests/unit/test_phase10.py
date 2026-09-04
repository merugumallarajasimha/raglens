"""Tests for query rewriting (Phase 10)."""

from __future__ import annotations

from app.retrieval.query_rewrite import QueryRewriter, RewriteResult, should_rewrite


class TestQueryRewriter:
    """Test the query rewriter."""

    def setup_method(self) -> None:
        self.rewriter = QueryRewriter(enable_multi_query=False)

    def test_rewrite_ambiguous_query(self) -> None:
        """Query with pronouns should be flagged as needing rewrite."""
        result = self.rewriter.rewrite("How does it improve RAG?")
        assert result.was_rewritten is True

    def test_rewrite_expands_rag_acronym(self) -> None:
        """RAG acronym should be expanded in rewritten query."""
        result = self.rewriter.rewrite("What is RAG?")
        assert "retrieval augmented generation" in result.rewritten_query.lower()

    def test_rewrite_preserves_simple_query(self) -> None:
        """Simple queries should not be unnecessarily rewritten."""
        result = self.rewriter.rewrite("What is hybrid retrieval?")
        assert result.original_query == "What is hybrid retrieval?"

    def test_rewrite_returns_original(self) -> None:
        """Original query should always be preserved."""
        original = "Compare Self-RAG and Corrective RAG"
        result = self.rewriter.rewrite(original)
        assert result.original_query == original

    def test_rewrite_with_context(self) -> None:
        """Query with 'this' should trigger rewrite flag."""
        result = self.rewriter.rewrite("What are the limitations of this paper?")
        assert result.was_rewritten is True

    def test_rewrite_multi_query_generates_variants(self) -> None:
        """Multi-query mode should generate expanded queries."""
        rewriter = QueryRewriter(enable_multi_query=True)
        result = rewriter.rewrite("What are the major approaches to RAG evaluation?")
        assert len(result.expanded_queries) > 1

    def test_rewrite_multi_query_dedupes(self) -> None:
        """Multi-query should not produce duplicate queries."""
        rewriter = QueryRewriter(enable_multi_query=True)
        result = rewriter.rewrite("Compare approaches in RAG")
        # No duplicates
        assert len(result.expanded_queries) == len(set(result.expanded_queries))

    def test_should_rewrite_simple_query(self) -> None:
        """Simple query without acronyms should not need rewriting."""
        assert should_rewrite("What is hybrid retrieval?") is False

    def test_should_rewrite_acronym(self) -> None:
        """Query with RAG acronym should need rewriting."""
        assert should_rewrite("How does RAG improve retrieval?") is True

    def test_should_rewrite_pronoun(self) -> None:
        """Query with pronouns should need rewriting."""
        assert should_rewrite("How does it work?") is True

    def test_should_rewrite_versus(self) -> None:
        """Queries with RAG acronym should be flagged for expansion."""
        result = should_rewrite("Compare Self-RAG and Corrective RAG")
        # Contains 'RAG' acronym which should be expanded
        assert result is True


class TestRewriteResult:
    """Test the RewriteResult dataclass."""

    def test_result_has_all_fields(self) -> None:
        rewriter = QueryRewriter()
        result = rewriter.rewrite("What is RAG?")
        assert hasattr(result, "original_query")
        assert hasattr(result, "rewritten_query")
        assert hasattr(result, "expanded_queries")
        assert hasattr(result, "was_rewritten")
        assert isinstance(result.expanded_queries, list)
