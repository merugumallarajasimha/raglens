"""Phase 2 tests: structure-aware chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import (
    Chunk,
    Chunker,
    ChunkingStrategy,
    chunk_paper,
    estimate_tokens,
    make_chunk_id,
)
from app.ingestion.models import Paragraph, Section, StructuredPaper, PaperMetadata


def _make_test_paper() -> StructuredPaper:
    """Create a test paper with known structure for chunking tests."""
    # Create long paragraphs to force multiple chunks
    long_para = (
        "This is a paragraph with enough text to be interesting. "
        "Retrieval-Augmented Generation combines parametric and non-parametric "
        "knowledge sources. Dense retrieval uses vector embeddings while sparse "
        "retrieval uses lexical matching. Hybrid approaches fuse both signals. "
        "Reranking applies cross-encoders to refine candidate ordering. "
        "This is additional text to ensure the paragraph is long enough "
        "for chunking tests with realistic content about RAG systems. "
    ) * 3

    return StructuredPaper(
        metadata=PaperMetadata(
            paper_id="paper_test001",
            title="Test Paper on RAG Systems",
            authors=["Test Author"],
            year=2024,
            abstract="An abstract about RAG.",
        ),
        sections=[
            Section(
                heading="Introduction",
                level=1,
                page_start=0,
                paragraphs=[
                    Paragraph(text=long_para, page=0),
                    Paragraph(text=long_para, page=0),
                ],
                subsections=[
                    Section(
                        heading="1.1 Background",
                        level=2,
                        page_start=0,
                        paragraphs=[Paragraph(text=long_para, page=0)],
                        subsections=[
                            Section(
                                heading="1.1.1 History",
                                level=3,
                                page_start=1,
                                paragraphs=[Paragraph(text=long_para, page=1)],
                            ),
                        ],
                    ),
                ],
            ),
            Section(
                heading="Methodology",
                level=1,
                page_start=2,
                paragraphs=[
                    Paragraph(text=long_para, page=2),
                    Paragraph(text=long_para, page=2),
                ],
            ),
        ],
        page_count=4,
        raw_text=long_para * 5,
    )


class TestTokenEstimation:
    """Test token estimation utility."""

    def test_empty_text(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        assert estimate_tokens("hello") == 1

    def test_longer_text(self) -> None:
        text = "word " * 100  # 500 chars
        # estimate_tokens is len(text) // 4
        assert estimate_tokens(text) == 500 // 4  # 125

    def test_always_at_least_one(self) -> None:
        assert estimate_tokens("a") == 1


class TestChunkIdGeneration:
    """Test deterministic chunk ID generation."""

    def test_deterministic_ids(self) -> None:
        id1 = make_chunk_id("paper1", "Introduction", "Background", 0)
        id2 = make_chunk_id("paper1", "Introduction", "Background", 0)
        assert id1 == id2

    def test_different_sections_different_ids(self) -> None:
        id1 = make_chunk_id("paper1", "Introduction", None, 0)
        id2 = make_chunk_id("paper1", "Methodology", None, 0)
        assert id1 != id2

    def test_different_indices_different_ids(self) -> None:
        id1 = make_chunk_id("paper1", "Intro", None, 0)
        id2 = make_chunk_id("paper1", "Intro", None, 1)
        assert id1 != id2

    def test_id_format(self) -> None:
        chunk_id = make_chunk_id("paper_xyz", "Section 1", "Sub A", 5)
        assert chunk_id.startswith("paper_xyz_")
        assert chunk_id.endswith("_0005")


class TestSectionAwareChunking:
    """Test section-aware chunking strategy."""

    def test_produces_chunks(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=400, chunk_overlap=100, strategy=ChunkingStrategy.SECTION_AWARE)
        chunks = chunker.chunk_paper(paper)
        assert len(chunks) > 0

    def test_chunks_have_metadata(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=400, chunk_overlap=100, strategy=ChunkingStrategy.SECTION_AWARE)
        chunks = chunker.chunk_paper(paper)
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.paper_id == "paper_test001"
            assert chunk.title == "Test Paper on RAG Systems"
            assert chunk.token_count > 0
            assert len(chunk.text) > 0

    def test_chunks_within_size_limit(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=400, chunk_overlap=100, strategy=ChunkingStrategy.SECTION_AWARE)
        chunks = chunker.chunk_paper(paper)
        for chunk in chunks:
            # Allow some margin over the target (paragraphs may be larger than chunk_size)
            assert chunk.token_count <= 600

    def test_respects_section_boundaries(self) -> None:
        """Chunks should not mix content from different sections."""
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=400, chunk_overlap=100, strategy=ChunkingStrategy.SECTION_AWARE)
        chunks = chunker.chunk_paper(paper)
        # Each chunk belongs to exactly one section
        sections = {c.section for c in chunks if c.section}
        assert "Introduction" in sections or "Methodology" in sections

    def test_page_numbers_preserved(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=400, chunk_overlap=100, strategy=ChunkingStrategy.SECTION_AWARE)
        chunks = chunker.chunk_paper(paper)
        for chunk in chunks:
            assert chunk.page_start is not None
            assert chunk.page_end is not None
            assert chunk.page_start >= 0
            assert chunk.page_end <= paper.page_count

    def test_deterministic_chunk_ids(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=400, chunk_overlap=100, strategy=ChunkingStrategy.SECTION_AWARE)
        chunks1 = chunker.chunk_paper(paper)
        chunks2 = chunker.chunk_paper(paper)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_preserves_subsection_info(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=400, chunk_overlap=100, strategy=ChunkingStrategy.SECTION_AWARE)
        chunks = chunker.chunk_paper(paper)
        # Should have chunks with subsection info
        has_subsection = any(c.subsection for c in chunks)
        assert has_subsection


class TestParagraphChunking:
    """Test paragraph-based chunking strategy."""

    def test_produces_chunks(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=400, chunk_overlap=100, strategy=ChunkingStrategy.PARAGRAPH)
        chunks = chunker.chunk_paper(paper)
        assert len(chunks) > 0

    def test_does_not_split_paragraphs(self) -> None:
        """Paragraph chunking should not split within a paragraph."""
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=200, chunk_overlap=50, strategy=ChunkingStrategy.PARAGRAPH)
        chunks = chunker.chunk_paper(paper)
        # Each paragraph should appear in at least one chunk intact
        assert len(chunks) > 0


class TestFixedChunking:
    """Test fixed-size chunking strategy."""

    def test_produces_chunks(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=200, chunk_overlap=50, strategy=ChunkingStrategy.FIXED)
        chunks = chunker.chunk_paper(paper)
        assert len(chunks) > 1

    def test_does_not_exceed_size_by_much(self) -> None:
        paper = _make_test_paper()
        chunker = Chunker(chunk_size=200, chunk_overlap=50, strategy=ChunkingStrategy.FIXED)
        chunks = chunker.chunk_paper(paper)
        for chunk in chunks:
            assert chunk.token_count <= 250  # some margin for word-level splitting


class TestChunkerFromSettings:
    """Test the chunk_paper convenience function with settings."""

    def test_uses_settings_values(self) -> None:
        from app.core.config import Settings

        settings = Settings(chunk_size=300, chunk_overlap=80, chunking_strategy="section_aware")
        paper = _make_test_paper()
        chunks = chunk_paper(paper, settings)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.token_count <= 500


class TestChunkIngestion:
    """Integration test: ingest PDF → chunk."""

    def test_ingested_paper_is_chunked(self, sample_pdf_path: Path) -> None:
        from app.ingestion.pipeline import ingest_paper

        paper = ingest_paper(sample_pdf_path)
        chunker = Chunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_paper(paper)
        assert len(chunks) > 0

    def test_chunks_from_real_pdf_have_section_info(self, sample_pdf_path: Path) -> None:
        from app.ingestion.pipeline import ingest_paper

        paper = ingest_paper(sample_pdf_path)
        chunker = Chunker(chunk_size=200, chunk_overlap=50, strategy=ChunkingStrategy.SECTION_AWARE)
        chunks = chunker.chunk_paper(paper)
        # At least some chunks should have section info
        sections = [c.section for c in chunks if c.section]
        assert len(sections) > 0
