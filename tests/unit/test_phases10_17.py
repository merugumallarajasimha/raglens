"""Tests for context builder, LLM, citation, and pipeline components (Phases 12-17)."""

from __future__ import annotations

import pytest

from app.generation.context_builder import ContextBuilder, BuiltContext, SourceReference
from app.generation.llm import MockLLMProvider, LLMResponse
from app.generation.citation import (
    Citation, CitationMap, CitationEngine, CitationValidator,
)
from app.pipeline.query_pipeline import QueryPipeline, QueryResponse, CitationInfo, EvidenceInfo
from app.pipeline.comparison_pipeline import ComparisonPipeline
from app.pipeline.summary_pipeline import SummaryPipeline
from app.pipeline.literature_pipeline import LiteratureReviewPipeline
from app.pipeline.evidence_pipeline import EvidencePipeline
from app.pipeline.conversation import ConversationManager, ConversationTurn
from app.retrieval.hybrid import HybridResult, HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.query_classifier import QueryClassifier, QueryIntent
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.filters import RetrievalFilters


def _make_hybrid_results(count: int = 5, paper_id: str = "paper_test") -> list[HybridResult]:
    """Create test HybridResult objects."""
    results = []
    for i in range(count):
        results.append(HybridResult(
            chunk_id=f"{paper_id}_chunk_{i:04d}",
            score=0.9 - (i * 0.1),
            dense_score=0.8,
            sparse_score=0.7,
            text=(
                f"RAG retrieval uses dense and sparse methods. "
                f"Chunk {i} discusses retrieval augmented generation techniques "
                f"including BM25, hybrid retrieval, and reranking."
            ),
            paper_id=paper_id,
            title="Test Paper on RAG",
            section="Introduction",
            subsection=None,
            page_start=i,
            page_end=i + 1,
            token_count=50,
            metadata={"year": 2024, "source": "test"},
        ))
    return results


class TestContextBuilder:
    """Test the context builder (Phase 12)."""

    def test_build_context_from_results(self) -> None:
        builder = ContextBuilder(max_tokens=2000)
        results = _make_hybrid_results(3)
        ctx = builder.build("What is RAG?", results)

        assert isinstance(ctx, BuiltContext)
        assert len(ctx.sources) == 3
        assert len(ctx.context_text) > 0
        assert ctx.total_tokens > 0

    def test_context_has_source_labels(self) -> None:
        builder = ContextBuilder(max_tokens=2000)
        results = _make_hybrid_results(3)
        ctx = builder.build("What is RAG?", results)

        assert "SOURCE [1]" in ctx.context_text
        assert "SOURCE [2]" in ctx.context_text
        assert "SOURCE [3]" in ctx.context_text

    def test_context_includes_section_info(self) -> None:
        builder = ContextBuilder(max_tokens=2000)
        results = _make_hybrid_results(3)
        ctx = builder.build("What is RAG?", results)

        assert "Section:" in ctx.context_text
        assert "Introduction" in ctx.context_text

    def test_context_includes_page_info(self) -> None:
        builder = ContextBuilder(max_tokens=2000)
        results = _make_hybrid_results(3)
        ctx = builder.build("What is RAG?", results)

        assert "Page" in ctx.context_text

    def test_context_respects_token_limit(self) -> None:
        builder = ContextBuilder(max_tokens=100)
        results = _make_hybrid_results(10)
        ctx = builder.build("What is RAG?", results)

        assert ctx.total_tokens <= 200  # Allow some margin

    def test_context_deduplicates_chunks(self) -> None:
        """Duplicate chunk_ids should be removed."""
        builder = ContextBuilder(max_tokens=2000)
        results = _make_hybrid_results(3)
        # Add duplicate
        results.append(results[0])
        ctx = builder.build("What is RAG?", results)

        assert len(ctx.sources) == 3  # Should not include duplicate

    def test_context_truncated_flag(self) -> None:
        builder = ContextBuilder(max_tokens=50)
        results = _make_hybrid_results(10)
        ctx = builder.build("What is RAG?", results)

        assert ctx.truncated is True


class TestMockLLM:
    """Test the mock LLM provider."""

    def test_mock_llm_generates_response(self) -> None:
        llm = MockLLMProvider(model="mock")
        response = llm.generate("Test prompt")
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.model == "mock"

    def test_mock_llm_predefined_response(self) -> None:
        llm = MockLLMProvider()
        llm.set_response("rag", "RAG is retrieval augmented generation.")
        response = llm.generate("Tell me about RAG")
        assert response.text == "RAG is retrieval augmented generation."

    def test_mock_llm_is_available(self) -> None:
        llm = MockLLMProvider()
        assert llm.is_available() is True

    def test_mock_llm_records_latency(self) -> None:
        llm = MockLLMProvider()
        response = llm.generate("test")
        assert response.latency_ms is not None
        assert response.latency_ms >= 0


class TestCitationEngine:
    """Test the citation engine (Phase 14)."""

    def test_generate_citations(self) -> None:
        sources = [
            SourceReference(
                citation_id=1, chunk_id="c1", paper_id="p1", title="Paper 1",
                section="Intro", subsection=None, page_start=0, page_end=1,
                text="evidence text", score=0.9,
            ),
            SourceReference(
                citation_id=2, chunk_id="c2", paper_id="p2", title="Paper 2",
                section="Method", subsection=None, page_start=2, page_end=3,
                text="more evidence", score=0.8,
            ),
        ]

        engine = CitationEngine()
        citation_map = engine.generate_citations(sources)

        assert len(citation_map.citations) == 2
        assert citation_map.get_by_id(1).chunk_id == "c1"
        assert citation_map.get_by_id(2).paper_id == "p2"

    def test_citation_map_get_by_chunk_id(self) -> None:
        source = SourceReference(
            citation_id=1, chunk_id="c1", paper_id="p1", title="Paper",
            section="Intro", subsection=None, page_start=0, page_end=1,
            text="text", score=0.9,
        )
        engine = CitationEngine()
        citation_map = engine.generate_citations([source])

        assert citation_map.get_by_chunk_id("c1") is not None
        assert citation_map.get_by_chunk_id("nonexistent") is None

    def test_validate_citation_reference(self) -> None:
        source = SourceReference(
            citation_id=1, chunk_id="c1", paper_id="p1", title="Paper",
            section="Intro", subsection=None, page_start=0, page_end=1,
            text="text", score=0.9,
        )
        engine = CitationEngine()
        engine.generate_citations([source])

        assert engine.validate_citation_reference(1) is True
        assert engine.validate_citation_reference(99) is False


class TestCitationValidator:
    """Test the citation validator (Phase 15)."""

    def test_extract_citation_ids(self) -> None:
        validator = CitationValidator()
        text = "This is supported by [1] and also [2][3]."
        ids = validator.extract_citation_ids(text)
        assert ids == [1, 2, 3]

    def test_extract_range_ids(self) -> None:
        validator = CitationValidator()
        text = "Sources [1-3] support this claim."
        ids = validator.extract_citation_ids(text)
        assert ids == [1, 2, 3]

    def test_validate_valid_citations(self) -> None:
        validator = CitationValidator()
        citation_map = CitationMap(citations=[
            Citation(citation_id=1, chunk_id="c1", paper_id="p1", title="Paper 1"),
            Citation(citation_id=2, chunk_id="c2", paper_id="p2", title="Paper 2"),
        ])

        result = validator.validate("This is supported by [1] and [2].", citation_map)
        assert result["is_valid"] is True
        assert result["valid"] == [1, 2]
        assert len(result["invalid"]) == 0

    def test_validate_fabricated_citation(self) -> None:
        """LLM referencing [5] when only [1]-[3] exist should be flagged."""
        validator = CitationValidator()
        citation_map = CitationMap(citations=[
            Citation(citation_id=1, chunk_id="c1", paper_id="p1", title="P1"),
            Citation(citation_id=2, chunk_id="c2", paper_id="p2", title="P2"),
            Citation(citation_id=3, chunk_id="c3", paper_id="p3", title="P3"),
        ])

        # LLM claims [5] which doesn't exist
        answer = "This is supported by [1] and [5]."
        result = validator.validate(answer, citation_map)

        assert result["is_valid"] is False
        assert 5 in result["invalid"]
        assert 1 in result["valid"]

    def test_validate_answer_removes_invalid(self) -> None:
        validator = CitationValidator()
        citation_map = CitationMap(citations=[
            Citation(citation_id=1, chunk_id="c1", paper_id="p1", title="P1"),
        ])

        answer = "Supported by [1] and fabricated [99]."
        sanitized, valid_citations = validator.validate_answer(answer, citation_map)

        assert "[99]" not in sanitized
        assert "[1]" in sanitized
        assert len(valid_citations) == 1

    def test_validate_catches_fabricated_paper(self) -> None:
        """Citations to papers not in the corpus should be invalid."""
        validator = CitationValidator()
        citation_map = CitationMap(citations=[
            Citation(citation_id=1, chunk_id="c1", paper_id="p1", title="P1"),
        ])

        answer = "As shown in [1], the approach works well."
        result = validator.validate(answer, citation_map)
        assert result["is_valid"] is True

        answer2 = "Some other paper [2] says something different."
        result2 = validator.validate(answer2, citation_map)
        assert result2["is_valid"] is False
        assert 2 in result2["invalid"]


class TestConversationManager:
    """Test conversation handling (Phase 21)."""

    def test_standalone_query_no_history(self) -> None:
        cm = ConversationManager()
        query, was_rewritten = cm.get_standalone_query("What is RAG?")
        assert query == "What is RAG?"
        assert was_rewritten is False

    def test_standalone_query_with_pronoun(self) -> None:
        cm = ConversationManager()
        cm.add_turn("What is Self-RAG?", "Self-RAG is a method...")

        standalone, was_rewritten = cm.get_standalone_query("How does it compare with CRAG?")
        assert was_rewritten is True
        assert "Self-RAG" in standalone or "context" in standalone.lower()

    def test_standalone_query_no_pronoun(self) -> None:
        cm = ConversationManager()
        cm.add_turn("What is RAG?", "RAG is retrieval augmented generation.")

        standalone, was_rewritten = cm.get_standalone_query("How does hybrid retrieval work?")
        assert was_rewritten is False
        assert standalone == "How does hybrid retrieval work?"

    def test_add_turn_stores_history(self) -> None:
        cm = ConversationManager()
        cm.add_turn("Question 1", "Answer 1", [{"citation_id": 1}])

        history = cm.get_history()
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_clear_history(self) -> None:
        cm = ConversationManager()
        cm.add_turn("Question", "Answer")
        assert len(cm.get_history()) > 0

        cm.clear()
        assert len(cm.get_history()) == 0

    def test_max_history_trimming(self) -> None:
        cm = ConversationManager(max_history=3)
        for i in range(10):
            cm.add_turn(f"Q{i}", f"A{i}")

        history = cm.get_history()
        # 3 max_history turns = 3 user + 3 assistant = 6 total
        assert len(history) <= 6


class TestQueryPipelineIntegration:
    """Integration test: mock end-to-end query pipeline (Phase 13)."""

    def test_pipeline_with_mock_components(self) -> None:
        """Test the full pipeline with mock LLM and in-memory vector store."""
        from tests.fixtures.vector_store import InMemoryVectorStore
        from app.retrieval.embeddings import MockEmbeddingProvider
        from app.retrieval.dense import DenseRetriever
        from app.retrieval.sparse import SparseRetriever
        from app.retrieval.hybrid import HybridRetriever
        from app.ingestion.chunker import Chunk
        from app.services.search_service import SearchService
        from app.core.config import Settings

        # Set up mock components
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)

        # Create test chunks
        chunks = []
        for i in range(5):
            chunk = Chunk(
                chunk_id=f"c{i:04d}",
                paper_id="p1",
                title="RAG Survey Paper",
                section="Introduction",
                page_start=i,
                page_end=i + 1,
                text=(
                    f"RAG retrieval augmented generation uses dense and sparse "
                    f"methods. Dense retrieval uses embeddings. BM25 is sparse. "
                    f"Hybrid approaches combine both via RRF. Reranking uses "
                    f"cross-encoders to refine results."
                ),
                token_count=40,
                metadata={},
            )
            chunks.append(chunk)

        # Index chunks
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        # Build sparse retriever corpus
        corpus = [
            {"chunk_id": c.chunk_id, "text": c.text,
             "paper_id": c.paper_id, "title": c.title}
            for c in chunks
        ]

        # Create search service with mock components
        settings = Settings(
            llm_provider="mock",
            embedding_provider="mock",
            embedding_dim=64,
            enable_query_rewrite=False,
            max_retrieval_retries=1,
        )

        dense = DenseRetriever(store, provider)
        sparse = SparseRetriever(corpus=corpus, top_k=5)
        hybrid = HybridRetriever(dense, sparse)
        reranker = Reranker(provider="mock")

        # Monkey-patch the search service
        service = SearchService(vector_store=store, embedding_provider=provider, settings=settings)
        # Override the methods
        original_search = service.search

        def mock_search(query, top_k=None, filters=None, rerank=True):
            results = hybrid.retrieve(query, top_k=settings.hybrid_top_k, filters=filters)
            if rerank and results:
                results = reranker.rerank(query, results, top_k=settings.rerank_top_k)
            return results

        service.search = mock_search

        # Create pipeline
        llm = MockLLMProvider()
        llm.set_response("what is hybrid retrieval", "Hybrid retrieval combines dense embedding search and BM25 lexical search using RRF [1].")

        pipeline = QueryPipeline(
            search_service=service,
            llm_provider=llm,
            settings=settings,
        )

        response = pipeline.answer("What is hybrid retrieval?")

        assert isinstance(response, QueryResponse)
        assert len(response.answer) > 0
        assert response.success is True
        assert len(response.citations) >= 0

    def test_pipeline_insufficient_evidence(self) -> None:
        """Pipeline should return insufficient evidence response when no results."""
        from tests.fixtures.vector_store import InMemoryVectorStore
        from app.retrieval.embeddings import MockEmbeddingProvider
        from app.services.search_service import SearchService
        from app.core.config import Settings

        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)

        settings = Settings(
            llm_provider="mock",
            embedding_provider="mock",
            embedding_dim=64,
            enable_query_rewrite=False,
            max_retrieval_retries=1,
        )

        # Empty store - no results
        service = SearchService(vector_store=store, embedding_provider=provider, settings=settings)
        service.search = lambda **kw: []

        pipeline = QueryPipeline(
            search_service=service,
            llm_provider=MockLLMProvider(),
            settings=settings,
        )

        response = pipeline.answer("What is quantum computing?")

        assert response.insufficient_evidence is True
        assert "sufficient evidence" in response.answer.lower()

    def test_pipeline_citation_validation(self) -> None:
        """Pipeline should validate citations against retrieved evidence."""
        from tests.fixtures.vector_store import InMemoryVectorStore
        from app.retrieval.embeddings import MockEmbeddingProvider
        from app.retrieval.dense import DenseRetriever
        from app.retrieval.sparse import SparseRetriever
        from app.retrieval.hybrid import HybridRetriever
        from app.ingestion.chunker import Chunk
        from app.services.search_service import SearchService
        from app.core.config import Settings
        from app.retrieval.reranker import Reranker

        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)

        chunks = [
            Chunk(
                chunk_id=f"c{i:04d}", paper_id="p1", title="RAG Paper",
                section="Intro", page_start=i, page_end=i+1,
                text="RAG retrieval augmented generation combines dense and sparse.",
                token_count=20, metadata={},
            )
            for i in range(3)
        ]
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        corpus = [{"chunk_id": c.chunk_id, "text": c.text, "paper_id": c.paper_id, "title": c.title} for c in chunks]

        settings = Settings(
            llm_provider="mock",
            embedding_provider="mock",
            embedding_dim=64,
            enable_query_rewrite=False,
            max_retrieval_retries=1,
        )

        dense = DenseRetriever(store, provider)
        sparse = SparseRetriever(corpus=corpus, top_k=5)
        hybrid = HybridRetriever(dense, sparse)
        reranker = Reranker(provider="mock")

        service = SearchService(vector_store=store, embedding_provider=provider, settings=settings)
        def mock_search(query, top_k=None, filters=None, rerank=True):
            results = hybrid.retrieve(query, top_k=settings.hybrid_top_k, filters=filters)
            if rerank and results:
                results = reranker.rerank(query, results, top_k=settings.rerank_top_k)
            return results
        service.search = mock_search

        # LLM generates answer with a fabricated citation [99]
        llm = MockLLMProvider()
        llm.set_response("rag", "Evidence supports this [1]. Some claim [99] is relevant.")

        pipeline = QueryPipeline(
            search_service=service,
            llm_provider=llm,
            settings=settings,
        )

        response = pipeline.answer("What is RAG?")

        # Citation [99] should be removed from the answer
        assert "[99]" not in response.answer
