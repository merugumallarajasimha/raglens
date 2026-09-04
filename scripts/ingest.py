"""Script: Ingest PDFs into RAGLens.

Usage:
    python scripts/ingest.py data/raw/paper1.pdf data/raw/paper2.pdf
    python scripts/ingest.py --dir data/raw/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from app.ingestion.pipeline import ingest_paper, ingest_papers_from_directory
from app.core.logging import setup_logging, get_logger


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

    if args.dir:
        papers = ingest_papers_from_directory(args.dir)
    elif args.files:
        papers = []
        for f in args.files:
            try:
                paper = ingest_paper(f)
                papers.append(paper)
            except Exception as e:
                logger.error(f"Failed to ingest {f}: {e}")
    else:
        parser.print_help()
        sys.exit(1)

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingested {len(papers)} paper(s)")

    for paper in papers:
        data = paper.model_dump()
        out_path = export_dir / f"{paper.metadata.paper_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [OK] {paper.metadata.title} ({paper.page_count} pages) -> {out_path}")

    print(f"\nDone: {len(papers)} paper(s) ingested to {export_dir}")


if __name__ == "__main__":
    main()
