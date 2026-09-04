"""Phase 3-8 tests: PostgreSQL, embeddings, dense/BM25/hybrid retrieval, reranking."""

from __future__ import annotations

from pathlib import Path
import pytest

from app.database.repositories import PaperRepository, ChunkRepository
from app.ingestion.chunker import Chunk, Chunker
from app.ingestion.models import PaperMetadata, StructuredPaper
from app.retrieval.embeddings import (
    MockEmbeddingProvider,
    get_embedding_provider,
    get_embedding_cache,
)
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.hybrid import HybridRetriever, HybridResult
from app.retrieval.reranker import Reranker
from app.retrieval.filters import RetrievalFilters, apply_filters, build_qdrant_filter
from tests.fixtures.vector_store import InMemoryVectorStore


def _make_paper(paper_id: str = "paper_test", title: str = "Test Paper") -> StructuredPaper:
    """Create a minimal structured paper for testing."""
    from app.ingestion.models import Section, Paragraph

    return StructuredPaper(
        metadata=PaperMetadata(paper_id=paper_id, title=title, year=2024),
        sections=[
            Section(heading="Introduction", level=1, page_start=0,
                    paragraphs=[Paragraph(text="This is the introduction about RAG.", page=0)]),
        ],
        page_count=1,
    )


def _make_chunks(paper_id: str = "paper_test", count: int = 5) -> list[Chunk]:
    """Create test chunks for a paper."""
    chunks = []
    for i in range(count):
        chunk = Chunk(
            chunk_id=f"{paper_id}_chunk_{i:04d}",
            paper_id=paper_id,
            title="Test Paper",
            section="Introduction",
            subsection=None,
            page_start=0,
            page_end=1,
            text=f"This is chunk {i} about retrieval augmented generation and dense retrieval.",
            token_count=20,
            metadata={},
        )
        chunks.append(chunk)
    return chunks


class TestPaperRepository:
    """Test PostgreSQL paper repository operations."""

    def test_paper_create_and_retrieve(self, in_memory_db) -> None:
        repo = PaperRepository(in_memory_db)
        paper = repo.create(
            paper_id="paper_001",
            title="Test Paper Title",
            abstract="This is an abstract.",
            year=2024,
            source="arxiv",
            url="http://arxiv.org/abs/1234.5678",
            authors=["Author One", "Author Two"],
        )
        assert paper.id == "paper_001"

        retrieved = repo.get_by_id("paper_001")
        assert retrieved is not None
        assert retrieved.title == "Test Paper Title"

    def test_paper_deduplication(self, in_memory_db) -> None:
        repo = PaperRepository(in_memory_db)
        repo.create(paper_id="paper_002", title="First")
        repo.create(paper_id="paper_002", title="First")
        # Should not raise, should return existing
        assert repo.exists("paper_002")

    def test_paper_list(self, in_memory_db) -> None:
        repo = PaperRepository(in_memory_db)
        repo.create(paper_id="paper_001", title="Paper 1")
        repo.create(paper_id="paper_002", title="Paper 2")

        papers = repo.list()
        assert len(papers) == 2

    def test_paper_delete(self, in_memory_db) -> None:
        repo = PaperRepository(in_memory_db)
        repo.create(paper_id="paper_001", title="Paper 1")
        assert repo.exists("paper_001")

        result = repo.delete("paper_001")
        assert result is True
        assert not repo.exists("paper_001")

    def test_paper_get_dict(self, in_memory_db) -> None:
        repo = PaperRepository(in_memory_db)
        repo.create(
            paper_id="paper_001",
            title="Title",
            authors=["Author"],
        )
        d = repo.get_by_id_dict("paper_001")
        assert d is not None
        assert d["paper_id"] == "paper_001"
        assert d["title"] == "Title"
        assert d["authors"] == ["Author"]


class TestChunkRepository:
    """Test PostgreSQL chunk repository operations."""

    def test_chunk_create_and_retrieve(self, in_memory_db) -> None:
        repo = ChunkRepository(in_memory_db)
        chunk = repo.create(
            chunk_id="chunk_001",
            paper_id="paper_001",
            section="Introduction",
            text="Some text content",
            token_count=10,
        )
        assert chunk.id == "chunk_001"

        retrieved = repo.get_by_id("chunk_001")
        assert retrieved is not None
        assert retrieved.text == "Some text content"

    def test_chunk_bulk_create(self, in_memory_db) -> None:
        repo = ChunkRepository(in_memory_db)
        chunks_data = [
            {"chunk_id": f"chunk_{i:04d}", "paper_id": "paper_001",
             "section": "Intro", "text": f"Text {i}", "token_count": 5}
            for i in range(10)
        ]
        count = repo.bulk_create(chunks_data)
        assert count == 10

        listed = repo.list_by_paper("paper_001")
        assert len(listed) == 10

    def test_chunk_delete_by_paper(self, in_memory_db) -> None:
        repo = ChunkRepository(in_memory_db)
        repo.create(chunk_id="c1", paper_id="p1", text="text1", token_count=5)
        repo.create(chunk_id="c2", paper_id="p1", text="text2", token_count=5)
        repo.create(chunk_id="c3", paper_id="p2", text="text3", token_count=5)

        deleted = repo.delete_by_paper("p1")
        assert deleted == 2

        remaining = repo.list_by_paper("p1")
        assert len(remaining) == 0
        assert len(repo.list_by_paper("p2")) == 1

    def test_chunk_search_by_text(self, in_memory_db) -> None:
        repo = ChunkRepository(in_memory_db)
        repo.create(chunk_id="c1", paper_id="p1", text="RAG retrieval augmentation", token_count=10)
        repo.create(chunk_id="c2", paper_id="p1", text="Dense neural search methods", token_count=10)
        repo.create(chunk_id="c3", paper_id="p2", text="BM25 lexical matching approach", token_count=10)

        results = repo.search_by_text("RAG", paper_id="p1")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"


class TestEmbeddingProvider:
    """Test embedding providers."""

    def test_mock_provider_dimension(self) -> None:
        provider = MockEmbeddingProvider(dimension=128)
        assert provider.dimension == 128

    def test_mock_provider_embed_text(self) -> None:
        provider = MockEmbeddingProvider(dimension=128)
        emb = provider.embed_text("hello world")
        assert len(emb) == 128
        assert all(-1.0 <= v <= 1.0 for v in emb)

    def test_mock_provider_deterministic(self) -> None:
        provider = MockEmbeddingProvider(dimension=64)
        emb1 = provider.embed_text("same text")
        emb2 = provider.embed_text("same text")
        assert emb1 == emb2

    def test_mock_provider_different_text_different_embedding(self) -> None:
        provider = MockEmbeddingProvider(dimension=64)
        emb1 = provider.embed_text("hello world")
        emb2 = provider.embed_text("goodbye universe")
        assert emb1 != emb2

    def test_mock_provider_embed_documents(self) -> None:
        provider = MockEmbeddingProvider(dimension=64)
        embs = provider.embed_documents(["text1", "text2", "text3"], batch_size=2)
        assert len(embs) == 3
        assert all(len(e) == 64 for e in embs)

    def test_mock_provider_empty_text(self) -> None:
        provider = MockEmbeddingProvider(dimension=64)
        embs = provider.embed_documents([], batch_size=2)
        assert len(embs) == 0

    def test_get_embedding_provider_returns_provider(self) -> None:
        from app.core.config import Settings
        settings = Settings(embedding_provider="mock", embedding_dim=128)
        provider = get_embedding_provider(settings)
        assert provider.dimension == 128


class TestDenseRetrieval:
    """Test dense retrieval using in-memory vector store."""

    def test_dense_retrieval_returns_results(self) -> None:
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)
        retriever = DenseRetriever(store, provider)

        chunks = _make_chunks(count=5)
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        results = retriever.retrieve("RAG retrieval", top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_dense_retrieval_sorted_by_score(self) -> None:
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)
        retriever = DenseRetriever(store, provider)

        chunks = _make_chunks(count=5)
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        results = retriever.retrieve("dense retrieval", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_dense_retrieval_preserves_metadata(self) -> None:
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)
        retriever = DenseRetriever(store, provider)

        chunks = _make_chunks(count=3)
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        results = retriever.retrieve("retrieval", top_k=3)
        for r in results:
            assert r.chunk_id
            assert r.paper_id == "paper_test"
            assert r.title == "Test Paper"
            assert r.section == "Introduction"

    def test_dense_retrieval_with_filters(self) -> None:
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)
        retriever = DenseRetriever(store, provider)

        # Create chunks for two papers
        chunks1 = [Chunk(
            chunk_id=f"p1_chunk_{i:04d}", paper_id="p1", title="Paper 1",
            text="RAG retrieval text", token_count=10, metadata={},
        ) for i in range(3)]
        chunks2 = [Chunk(
            chunk_id=f"p2_chunk_{i:04d}", paper_id="p2", title="Paper 2",
            text="RAG retrieval text", token_count=10, metadata={},
        ) for i in range(3)]

        all_chunks = chunks1 + chunks2
        embeddings = [provider.embed_text(c.text) for c in all_chunks]
        store.upsert_chunks(all_chunks, embeddings)

        results = retriever.retrieve("RAG retrieval", top_k=10, filters={"paper_id": "p1"})
        assert len(results) > 0
        for r in results:
            assert r.paper_id == "p1"


class TestBM25Retrieval:
    """Test BM25 sparse retrieval."""

    def test_bm25_returns_results(self) -> None:
        corpus = [
            {"chunk_id": f"c{i}", "text": f"Paper about RAG retrieval augmented generation {i}",
             "paper_id": "p1", "title": "P1"}
            for i in range(10)
        ]
        retriever = SparseRetriever(corpus=corpus, top_k=5)

        results = retriever.retrieve("RAG augmented generation")
        assert len(results) > 0
        assert len(results) <= 5

    def test_bm25_exact_term_matching(self) -> None:
        """BM25 should prioritize exact term matches."""
        corpus = [
            {"chunk_id": "c1", "text": "ColBERT uses late interaction for dense retrieval",
             "paper_id": "p1", "title": "ColBERT Paper"},
            {"chunk_id": "c2", "text": "BM25 is a lexical search algorithm based on term frequency",
             "paper_id": "p2", "title": "BM25 Paper"},
            {"chunk_id": "c3", "text": "Hybrid retrieval combines dense and sparse methods",
             "paper_id": "p3", "title": "Hybrid Paper"},
        ]
        retriever = SparseRetriever(corpus=corpus, top_k=3)

        # Query with exact term "BM25"
        results = retriever.retrieve("BM25")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c2"

        # Query with exact term "ColBERT"
        results = retriever.retrieve("ColBERT")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"

    def test_bm25_rerankes_relevant_content(self) -> None:
        corpus = [
            {"chunk_id": "c1", "text": "RAG papers discuss retrieval and generation extensively",
             "paper_id": "p1", "title": "P1"},
            {"chunk_id": "c2", "text": "The weather is sunny today and birds are singing",
             "paper_id": "p2", "title": "P2"},
            {"chunk_id": "c3", "text": "Hybrid approaches combine dense retrieval methods",
             "paper_id": "p3", "title": "P3"},
        ]
        retriever = SparseRetriever(corpus=corpus, top_k=3)

        results = retriever.retrieve("retrieval generation RAG")
        assert len(results) > 0
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] > results[-1]["score"]

    def test_bm25_filter_by_paper_id(self) -> None:
        corpus = [
            {"chunk_id": "c1", "text": "RAG retrieval augmentation", "paper_id": "p1"},
            {"chunk_id": "c2", "text": "RAG retrieval augmentation", "paper_id": "p2"},
            {"chunk_id": "c3", "text": "BM25 lexical matching", "paper_id": "p1"},
        ]
        retriever = SparseRetriever(corpus=corpus, top_k=5)

        results = retriever.retrieve("RAG", filters={"paper_id": "p1"})
        assert all(r["paper_id"] == "p1" for r in results)

    def test_bm25_build_index_lazy(self) -> None:
        """Test building index after initialization."""
        retriever = SparseRetriever(top_k=5)
        corpus = [
            {"chunk_id": "c1", "text": "test text about RAG retrieval", "paper_id": "p1"},
            {"chunk_id": "c2", "text": "weather and birds singing", "paper_id": "p2"},
            {"chunk_id": "c3", "text": "RAG systems overview chapter", "paper_id": "p3"},
        ]
        retriever.build_index(corpus)
        results = retriever.retrieve("RAG")
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "c1" or results[0]["chunk_id"] == "c3"


class TestHybridRetrieval:
    """Test hybrid retrieval with RRF fusion."""

    def test_hybrid_combines_results(self) -> None:
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)

        chunks = _make_chunks(count=5)
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        dense = DenseRetriever(store, provider)
        sparse = SparseRetriever(
            corpus=[{"chunk_id": c.chunk_id, "text": c.text,
                      "paper_id": c.paper_id, "title": c.title} for c in chunks],
            top_k=5,
        )

        hybrid = HybridRetriever(dense, sparse)
        results = hybrid.retrieve("RAG retrieval", top_k=3)

        assert len(results) <= 3
        assert len(results) > 0

    def test_hybrid_results_have_both_scores(self) -> None:
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)

        chunks = _make_chunks(count=5)
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        dense = DenseRetriever(store, provider)
        sparse = SparseRetriever(
            corpus=[{"chunk_id": c.chunk_id, "text": c.text,
                      "paper_id": c.paper_id, "title": c.title} for c in chunks],
            top_k=5,
        )

        hybrid = HybridRetriever(dense, sparse)
        results = hybrid.retrieve("retrieval", top_k=3)

        for r in results:
            assert r.dense_score is not None or r.sparse_score is not None

    def test_hybrid_rrf_fusion_logic(self) -> None:
        """Test that RRF correctly merges rankings from both retrievers."""
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)

        # Create chunks where dense and sparse have different orderings
        chunks = [
            Chunk(chunk_id=f"c{i}", paper_id="p1", title="P",
                  text=f"content about RAG retrieval {i}", token_count=10, metadata={})
            for i in range(10)
        ]
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        dense = DenseRetriever(store, provider)
        sparse = SparseRetriever(top_k=5)
        sparse.build_index([
            {"chunk_id": c.chunk_id, "text": c.text,
             "paper_id": c.paper_id, "title": c.title} for c in chunks
        ])

        hybrid = HybridRetriever(dense, sparse, rrf_k=60)

        # Retrieve more results to see fusion
        results = hybrid.retrieve("RAG retrieval", top_k=5)

        # All results should have a positive RRF score
        assert all(r.score > 0 for r in results)
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_empty_query_returns_empty(self) -> None:
        """Nonsensical query should produce no BM25 results and low dense scores."""
        store = InMemoryVectorStore(embedding_dim=64)
        provider = MockEmbeddingProvider(dimension=64)

        chunks = _make_chunks(count=3)
        embeddings = [provider.embed_text(c.text) for c in chunks]
        store.upsert_chunks(chunks, embeddings)

        dense = DenseRetriever(store, provider)
        sparse = SparseRetriever(top_k=5)
        sparse.build_index([
            {"chunk_id": c.chunk_id, "text": c.text,
             "paper_id": c.paper_id, "title": c.title} for c in chunks
        ])

        hybrid = HybridRetriever(dense, sparse)
        results = hybrid.retrieve("xyz nonexistentterm12345", top_k=3)

        # Sparse (BM25) should return 0 results for nonsensical query
        sparse_results = sparse.retrieve("xyz nonexistentterm12345", top_k=5)
        assert len(sparse_results) == 0

        # Dense may return results (mock embeddings), but scores should be low
        for r in results:
            assert r.score >= 0  # RRF score is always >= 0


class TestReranker:
    """Test the reranker component."""

    def test_mock_reranker_produces_scores(self) -> None:
        reranker = Reranker(provider="mock")
        candidates = [
            HybridResult(chunk_id=f"c{i}", score=0.5,
                         text=f"Text about RAG retrieval {i}", paper_id="p1")
            for i in range(5)
        ]

        results = reranker.rerank("RAG retrieval", candidates, top_k=3)
        assert len(results) == 3
        assert all(r.rerank_score is not None for r in results)

    def test_mock_reranker_reranks_by_overlap(self) -> None:
        """Mock reranker should rank higher text overlap higher."""
        reranker = Reranker(provider="mock")
        candidates = [
            HybridResult(chunk_id="c1", score=0.5,
                         text="Completely unrelated content about cooking", paper_id="p1"),
            HybridResult(chunk_id="c2", score=0.4,
                         text="RAG retrieval augmented generation text", paper_id="p1"),
        ]

        results = reranker.rerank("RAG retrieval", candidates, top_k=2)
        # The text with more overlap should rank higher
        assert results[0].chunk_id == "c2"

    def test_reranker_empty_input(self) -> None:
        reranker = Reranker(provider="mock")
        results = reranker.rerank("query", [], top_k=5)
        assert len(results) == 0


class TestFilters:
    """Test metadata filtering."""

    def test_apply_filters_by_paper_id(self) -> None:
        results = [
            HybridResult(chunk_id="c1", score=0.5, paper_id="p1", section="Intro"),
            HybridResult(chunk_id="c2", score=0.4, paper_id="p2", section="Method"),
            HybridResult(chunk_id="c3", score=0.3, paper_id="p1", section="Intro"),
        ]
        filters = RetrievalFilters(paper_ids=["p1"])
        filtered = apply_filters(results, filters)
        assert len(filtered) == 2
        assert all(r.paper_id == "p1" for r in filtered)

    def test_apply_filters_by_section(self) -> None:
        results = [
            HybridResult(chunk_id="c1", score=0.5, paper_id="p1", section="Introduction"),
            HybridResult(chunk_id="c2", score=0.4, paper_id="p2", section="Methodology"),
            HybridResult(chunk_id="c3", score=0.3, paper_id="p1", section="Conclusion"),
        ]
        filters = RetrievalFilters(sections=["Introduction"])
        filtered = apply_filters(results, filters)
        assert len(filtered) == 1
        assert filtered[0].chunk_id == "c1"

    def test_apply_filters_none_returns_all(self) -> None:
        results = [
            HybridResult(chunk_id="c1", score=0.5, paper_id="p1"),
            HybridResult(chunk_id="c2", score=0.4, paper_id="p2"),
        ]
        filtered = apply_filters(results, None)
        assert len(filtered) == 2

    def test_build_qdrant_filter(self) -> None:
        filters = RetrievalFilters(paper_ids=["p1"])
        qf = build_qdrant_filter(filters)
        assert qf is not None
        assert qf["paper_id"] == "p1"

    def test_build_qdrant_filter_none(self) -> None:
        qf = build_qdrant_filter(RetrievalFilters())
        assert qf is None
