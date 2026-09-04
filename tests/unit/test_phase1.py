"""Phase 1 tests: PDF ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.exceptions import PDFParseError
from app.ingestion.models import StructuredPaper
from app.ingestion.pdf_parser import compute_paper_id, parse_pdf, extract_pdf_metadata
from app.ingestion.metadata_extractor import (
    extract_authors,
    extract_abstract,
    extract_year,
    extract_metadata,
)
from app.ingestion.structure_parser import parse_structure
from app.ingestion.pipeline import ingest_paper


class TestPdfParser:
    """Test the PDF parser."""

    def test_parse_pdf_returns_structured_paper(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        assert isinstance(paper, StructuredPaper)
        assert paper.page_count == 4
        assert paper.metadata.paper_id.startswith("paper_")
        assert paper.raw_text is not None
        assert len(paper.raw_text) > 100

    def test_parse_pdf_preserves_page_count(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        assert paper.page_count == 4

    def test_parse_pdf_preserves_raw_text(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        raw = paper.raw_text
        assert "Abstract" in raw
        assert "Introduction" in raw
        assert "References" in raw
        assert "Retrieval-Augmented Generation" in raw

    def test_compute_paper_id_deterministic(self, sample_pdf_path: Path) -> None:
        id1 = compute_paper_id(sample_pdf_path)
        id2 = compute_paper_id(sample_pdf_path)
        assert id1 == id2
        assert id1.startswith("paper_")

    def test_compute_paper_id_different_files_different_ids(
        self, sample_pdf_path: Path, sample_pdf_path_2: Path
    ) -> None:
        id1 = compute_paper_id(sample_pdf_path)
        id2 = compute_paper_id(sample_pdf_path_2)
        assert id1 != id2

    def test_parse_pdf_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/file.pdf")

    def test_parse_pdf_extracts_metadata_fields(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        pdf_meta = getattr(paper, "_pdf_meta", {})
        assert "title" in pdf_meta or "author" in pdf_meta

    def test_pages_have_span_info(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        pages = getattr(paper, "_pages", [])
        assert len(pages) == paper.page_count
        first_page = pages[0]
        assert len(first_page.lines) > 0
        # Check that spans have font size info
        for line in first_page.lines[:5]:
            if line.spans:
                assert line.max_font_size > 0
                assert line.page == 0


class TestMetadataExtractor:
    """Test metadata extraction."""

    def test_extract_abstract_finds_text(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        abstract = extract_abstract(paper)
        assert abstract is not None
        assert len(abstract) > 50
        assert "retrieval-augmented generation" in abstract.lower()

    def test_extract_abstract_contains_key_terms(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        abstract = extract_abstract(paper)
        assert abstract is not None
        assert "dense" in abstract.lower() or "sparse" in abstract.lower()

    def test_extract_year_from_content(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        year = extract_year(paper)
        assert year == 2024

    def test_extract_year_second_paper(self, sample_pdf_path_2: Path) -> None:
        paper = parse_pdf(sample_pdf_path_2)
        year = extract_year(paper)
        assert year == 2023

    def test_extract_authors_returns_list(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        authors = extract_authors(paper)
        assert isinstance(authors, list)

    def test_extract_metadata_full_pipeline(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        paper = extract_metadata(paper)
        paper = parse_structure(paper)

        assert paper.metadata.title is not None
        assert len(paper.metadata.title) > 0
        assert paper.metadata.abstract is not None
        assert paper.metadata.year == 2024
        assert isinstance(paper.metadata.authors, list)
        assert paper.metadata.references is not None
        assert len(paper.metadata.references) > 0


class TestStructureParser:
    """Test structure parsing."""

    def test_parse_structure_creates_sections(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        paper = parse_structure(paper)

        assert len(paper.sections) > 0
        # Should have at least Introduction and Methodology sections
        all_headings = [s.heading for s in paper.sections]
        all_text = " ".join(all_headings).lower()
        assert "introduction" in all_text or "1" in all_text

    def test_parse_structure_has_page_numbers(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        paper = parse_structure(paper)

        for section in paper.sections:
            assert section.page_start >= 0
            for para in section.paragraphs:
                assert para.page >= 0

    def test_parse_structure_preserves_hierarchy(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        paper = parse_structure(paper)

        # Should have subsections (e.g., "2.1 Dense Retrieval")
        total_subsections = sum(len(s.subsections) for s in paper.sections)
        assert total_subsections > 0

    def test_parse_structure_section_has_level(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        paper = parse_structure(paper)

        for section in paper.sections:
            assert section.level >= 1
            for sub in section.subsections:
                assert sub.level >= 2

    def test_parse_structure_cleans_up_pages(self, sample_pdf_path: Path) -> None:
        paper = parse_pdf(sample_pdf_path)
        assert hasattr(paper, "_pages")
        paper = parse_structure(paper)
        assert not hasattr(paper, "_pages")

    def test_parse_structure_fallback_when_no_headings(self) -> None:
        """When no headings are detected, content should be in a fallback section."""
        from app.ingestion.models import PaperMetadata, StructuredPaper
        from app.ingestion.pdf_parser import TextSpan, TextLine, PageText

        paper = StructuredPaper(
            metadata=PaperMetadata(paper_id="test_paper", title="Test"),
            sections=[],
            page_count=1,
            raw_text="Some body text without headings.",
        )

        span = TextSpan(
            text="Just some text here.",
            page_no=0,
            font_size=11.0,
            font_name="helv",
            is_bold=False,
            bbox_top=100,
            bbox_bottom=110,
        )
        line = TextLine(spans=[span])
        page = PageText(page_no=0, lines=[line])

        paper._pages = [page]
        paper = parse_structure(paper)

        assert len(paper.sections) >= 1
        assert len(paper.sections[0].paragraphs) >= 1


class TestIngestPipeline:
    """Test the full ingestion pipeline."""

    def test_ingest_paper_returns_complete_paper(self, sample_pdf_path: Path) -> None:
        paper = ingest_paper(sample_pdf_path)

        assert paper.metadata.paper_id.startswith("paper_")
        assert paper.metadata.title is not None
        assert paper.metadata.abstract is not None
        assert paper.metadata.year == 2024
        assert len(paper.sections) > 0
        assert paper.page_count == 4

    def test_ingest_paper_has_references(self, sample_pdf_path: Path) -> None:
        paper = ingest_paper(sample_pdf_path)
        assert len(paper.metadata.references) > 0
        assert "Retrieval-Augmented Generation" in paper.metadata.references[0] or \
               "Lewis" in paper.metadata.references[0]

    def test_ingest_paper_sections_contain_paragraphs(self, sample_pdf_path: Path) -> None:
        paper = ingest_paper(sample_pdf_path)
        total_paras = sum(
            len(s.paragraphs) + sum(len(sub.paragraphs) for sub in s.subsections)
            for s in paper.sections
        )
        assert total_paras > 0

    def test_ingest_paper_preserves_page_info(self, sample_pdf_path: Path) -> None:
        paper = ingest_paper(sample_pdf_path)
        # Check that paragraphs have valid page numbers
        for section in paper.sections:
            for para in section.paragraphs:
                assert para.page >= 0
                assert para.page < paper.page_count
            for sub in section.subsections:
                for para in sub.paragraphs:
                    assert para.page >= 0
                    assert para.page < paper.page_count

    def test_ingest_paper_serializable(self, sample_pdf_path: Path) -> None:
        """Ingested paper should be serializable to JSON/dict."""
        paper = ingest_paper(sample_pdf_path)
        data = paper.model_dump()
        json_str = json.dumps(data)
        assert len(json_str) > 0
        assert "metadata" in data
        assert "sections" in data
