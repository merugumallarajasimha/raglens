"""Generate synthetic test PDFs that mimic real research paper structure."""

from __future__ import annotations

from pathlib import Path

import pymupdf

_TITLE_STYLE = {"fontname": "helv", "fontsize": 20}
_AUTHOR_STYLE = {"fontname": "helv", "fontsize": 12}
_ABSTRACT_HEADING_STYLE = {"fontname": "hebo", "fontsize": 14}
_SECTION_HEADING_STYLE = {"fontname": "hebo", "fontsize": 16}
_SUBSECTION_HEADING_STYLE = {"fontname": "hebo", "fontsize": 13}
_BODY_STYLE = {"fontname": "helv", "fontsize": 11}

_BODY_TEXT = """\
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for
combining the parametric knowledge of large language models with non-parametric
external knowledge sources. RAG models generate output by first retrieving
relevant documents from a corpus and then conditioning the language model on
the retrieved information.

Dense retrieval models like DPR (Dense Passage Retrieval) encode queries and
passages into a shared dense vector space and retrieve nearest neighbors using
maximum inner product search. BM25, on the other hand, is a sparse retrieval
method based on term frequency statistics.

Hybrid retrieval combines the strengths of both dense and sparse approaches.
Dense retrieval captures semantic similarity while sparse retrieval excels at
exact term matching. Reciprocal Rank Fusion (RRF) is a principled method for
combining these approaches into a unified candidate set.

Reranking further improves retrieval quality by using cross-encoders that jointly
encode the query and each candidate passage. This approach considers fine-grained
interactions between the query and each candidate but is computationally more
expensive than bi-encoder approaches.

Self-RAG is a variant that incorporates self-reflection into the RAG pipeline.
It generates reflection tokens that allow the model to critique its own
retrieval and generation steps, leading to more grounded responses.
"""


def _insert_heading(page: pymupdf.Page, text: str, y: float, style: dict) -> float:
    """Insert a heading and return the y position after it."""
    page.insert_text((72, y), text, **style)
    return y + style["fontsize"] + 10


def _insert_paragraph(page: pymupdf.Page, text: str, y: float, width: float = 380) -> float:
    """Insert a paragraph with wrapping and return the y position after it."""
    import textwrap

    paragraphs = text.strip().split("\n\n")
    for para in paragraphs:
        wrapped = textwrap.wrap(para.strip(), width=100)
        for line in wrapped:
            if y > 760:
                return y
            page.insert_text((72, y), line, **_BODY_STYLE)
            y += _BODY_STYLE["fontsize"] + 3
        y += 5  # paragraph spacing
    return y


def _insert_numbered_paragraph(page: pymupdf.Page, text: str, start_num: int, y: float) -> tuple[float, int]:
    """Insert numbered paragraphs like '1. text' and return (y, next_num)."""
    import textwrap

    num = start_num
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if y > 760:
            break
        # Wrap long lines
        wrapped = textwrap.wrap(line, width=90)
        for i, w in enumerate(wrapped):
            label = f"{num}." if i == 0 else "   "
            page.insert_text((72, y), f"{label} {w}", **_BODY_STYLE)
            y += _BODY_STYLE["fontsize"] + 3
            if i > 0:
                num += 1
        num += 1
    return y, num


def create_test_paper_pdf(
    output_path: str | Path,
    title: str = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    authors: str = "J. Smith, A. Johnson, M. Chen",
    year: int = 2024,
) -> str:
    """Create a synthetic research paper PDF for testing.

    The PDF has realistic structure:
    - Title
    - Authors
    - Abstract section
    - Numbered sections (1, 2, 3) with subsections (1.1, 1.2, etc.)
    - Body paragraphs within each section
    - References section

    Args:
        output_path: Where to save the PDF.
        title: Paper title.
        authors: Author string.
        year: Publication year.

    Returns:
        Path to the created PDF.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()

    # --- Page 1: Title, authors, abstract ---
    page = doc.new_page(width=612, height=792)  # US Letter

    y = 150
    page.insert_text((72, y), title, **_TITLE_STYLE)
    y += 30
    page.insert_text((72, y), authors, **_AUTHOR_STYLE)
    y += 25
    page.insert_text((72, y), f"Published: {year}", **_AUTHOR_STYLE)
    y += 40

    # Abstract section
    y = _insert_heading(page, "Abstract", y, _ABSTRACT_HEADING_STYLE)
    y += 5

    abstract_text = (
        "We present a comprehensive study of retrieval-augmented generation (RAG) "
        "systems. Our work examines the trade-offs between dense and sparse retrieval "
        "methods, and demonstrates that combining them via reciprocal rank fusion "
        "significantly improves retrieval quality. We evaluate multiple reranking "
        "strategies and show their impact on downstream generation quality. Our "
        "experiments cover dense retrieval models like DPR, sparse methods like "
        "BM25, and hybrid approaches. We also introduce Self-RAG, a method that "
        "incorporates self-reflection to improve grounding."
    )
    y = _insert_paragraph(page, abstract_text, y)

    # --- Page 2: Sections ---
    page = doc.new_page(width=612, height=792)
    y = 72

    y = _insert_heading(page, "1 Introduction", y, _SECTION_HEADING_STYLE)
    y += 5
    y = _insert_paragraph(page, _BODY_TEXT, y)

    # --- Page 3: More sections ---
    page = doc.new_page(width=612, height=792)
    y = 72

    y = _insert_heading(page, "2 Related Work", y, _SECTION_HEADING_STYLE)
    y += 5
    related_text = (
        "2.1 Dense Retrieval\n\nDPR encodes queries and passages using separate "
        "bi-encoders and retrieves the most similar passages via vector similarity. "
        "This approach captures semantic relationships but may miss exact term matches.\n\n"
        "2.2 Sparse Retrieval\n\nBM25 is a classical sparse retrieval method based on "
        "term frequency and inverse document frequency statistics. It excels at exact "
        "term matching but lacks semantic understanding.\n\n"
        "2.3 Hybrid Approaches\n\nRecent work has explored combining dense and sparse "
        "retrieval. RRF provides a principled way to fuse ranking lists from multiple "
        "retrievers without tuning combination parameters."
    )
    y = _insert_paragraph(page, related_text, y)

    # --- Page 4: Method + References ---
    page = doc.new_page(width=612, height=792)
    y = 72

    y = _insert_heading(page, "3 Methodology", y, _SECTION_HEADING_STYLE)
    y += 5
    y = _insert_heading(page, "3.1 Hybrid Retrieval Pipeline", y, _SUBSECTION_HEADING_STYLE)
    y += 5
    method_text = (
        "Our hybrid retrieval pipeline first retrieves top-k candidates using both "
        "dense and sparse retrievers. We then apply reciprocal rank fusion to combine "
        "the results into a unified candidate set. A cross-encoder reranker then "
        "re-scores the candidates based on query-candidate interaction.\n\n"
        "3.2 Reranking\n\nThe reranker uses a cross-encoder architecture that jointly "
        "encodes the query and each candidate passage. This allows the model to capture "
        "fine-grained relevance signals that are not available in bi-encoder retrieval."
    )
    y = _insert_paragraph(page, method_text, y)

    # References
    y = _insert_heading(page, "References", y + 15, _SECTION_HEADING_STYLE)
    y += 5
    refs = [
        "1. Lewis, P., et al. 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.' NeurIPS 2020.",
        "2. Karpuk, J. 'ColBERT: Efficient and Effective Passage Search via Late Interaction.' SIGIR 2020.",
        "3. Nogueira, R. and Cho, K. 'Passage Re-ranking with BERT.' SIGIR 2020.",
        "4. Gao, L., et al. 'Real-time Retrieval-Augmented Language Models.' ICML 2023.",
    ]
    for ref in refs:
        if y > 760:
            page = doc.new_page(width=612, height=792)
            y = 72
        page.insert_text((72, y), ref, **_BODY_STYLE)
        y += _BODY_STYLE["fontsize"] + 6

    doc.set_metadata({
        "title": title,
        "author": authors,
        "subject": f"A study on RAG and retrieval-augmented generation ({year})",
        "creationDate": f"D:{year}0101000000",
    })

    doc.save(str(path), garbage=4, deflate=True)
    doc.close()

    return str(path)
