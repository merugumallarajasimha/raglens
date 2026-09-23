"""Script: Ingest PDFs into RAGLens.

Usage:
    python scripts/ingest.py data/raw/Attention-is-all-you-need.pdf
    python scripts/ingest.py --dir data/raw/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the Python path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from app.ingestion.pipeline import ingest_paper, ingest_papers_from_directory
from app.database.vector_store import QdrantVectorStore
from app.retrieval.embeddings import get_embedding_provider
from app.retrieval.sparse import SparseRetriever
from app.core.logging import setup_logging, get_logger
from app.core.config import get_settings


def main() -> None:
    setup_logging()
    logger = get_logger("scripts.ingest")

    parser = argparse.ArgumentParser(description="Ingest PDFs into RAGLens")
    parser.add_argument("files", nargs="*", help="PDF file paths to ingest")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory to ingest all PDFs from (recursive)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="data/processed",
        help="Directory to export structured JSON (default: data/processed)",
    )
    args = parser.parse_args()

    papers = []

    if args.dir:
        dir_path = Path(args.dir)
        logger.info(f"Ingesting PDFs from directory: {dir_path}")
        papers = ingest_papers_from_directory(str(dir_path))
    elif args.files:
        for f in args.files:
            file_path = Path(f)
            try:
                logger.info(f"Processing PDF: {file_path}")
                paper = ingest_paper(str(file_path))
                papers.append(paper)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
    else:
        parser.print_help()
        sys.exit(1)

    if not papers:
        logger.warning("No papers were successfully processed.")
        sys.exit(1)

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Vector Store and Sparse Indexer
    logger.info("Indexing chunks into Qdrant vector database...")
    try:
        settings = get_settings()
        vector_store = QdrantVectorStore.from_settings(settings)
        embedding_provider = get_embedding_provider(settings)
        sparse_retriever = SparseRetriever()

        # Collect all chunks across papers
        all_chunks = []
        for paper in papers:
            if hasattr(paper, "chunks") and paper.chunks:
                all_chunks.extend(paper.chunks)

        # Build BM25 sparse index from the full corpus
        corpus = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "paper_id": c.paper_id,
                "title": c.title,
                "section": c.section,
                "subsection": c.subsection,
                "page_start": c.page_start,
                "page_end": c.page_end,
            }
            for c in all_chunks
        ]
        sparse_retriever.build_index(corpus)
        logger.info(f"BM25 index built with {len(corpus)} documents")

        # Generate embeddings and upsert to Qdrant
        embeddings = embedding_provider.embed_documents([c.text for c in all_chunks])
        vector_store.upsert_chunks(all_chunks, embeddings)
        logger.info(f"Indexed {len(all_chunks)} chunks for {len(papers)} paper(s)")
    except Exception as e:
        logger.error(f"Failed to index paper chunks into vector database: {e}")
        raise

    logger.info(f"Exporting structured JSON for {len(papers)} paper(s)...")

    for paper in papers:
        # Pydantic v2 / v1 fallback serialization
        if hasattr(paper, "model_dump"):
            data = paper.model_dump()
        else:
            data = paper.dict()

        out_path = export_dir / f"{paper.metadata.paper_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [OK] {paper.metadata.title} ({paper.page_count} pages) -> {out_path}")

    print(f"\nDone: {len(papers)} paper(s) ingested and indexed into Qdrant successfully!")


if __name__ == "__main__":
    main()