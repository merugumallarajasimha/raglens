"""Sparse retrieval using BM25 (lexical matching).

Operates independently from dense retrieval. Uses the rank-bm25 library
to build an in-memory BM25 index over chunk text.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
import re

from rank_bm25 import BM25Okapi

from app.core.exceptions import RetrievalError

logger = logging.getLogger("raglens.retrieval.sparse")


class SparseRetriever:
    """BM25-based sparse retriever for lexical search.

    Args:
        top_k: Default number of results to return.
    """

    def __init__(
        self,
        corpus: Optional[list[dict]] = None,
        top_k: int = 20,
    ) -> None:
        self._top_k = top_k
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._bm25 = None
        self._tokenized_corpus: list[list[str]] = []

        if corpus:
            self.build_index(corpus)

    def build_index(self, corpus: list[dict]) -> None:
        """Build the BM25 index from a corpus of chunk documents.

        Each document should be a dict with:
            - chunk_id: str
            - text: str
            - paper_id, title, section, page_start, page_end (optional)
        """
        self._chunk_ids = []
        self._texts = []
        self._metadata = []
        self._tokenized_corpus = []

        for doc in corpus:
            text = doc.get("text", "")
            if not text.strip():
                continue
            self._chunk_ids.append(doc["chunk_id"])
            self._texts.append(text)
            self._metadata.append(doc)
            self._tokenized_corpus.append(self._tokenize(text))

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(
            "BM25 index built",
            extra={"extra_data": {
                "documents": len(self._chunk_ids),
                "vocabulary_size": self._estimate_vocabulary_size(),
            }},
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Lowercases, splits on word boundaries, keeps technical terms like
        model names and acronyms.
        """
        # Lowercase and split on non-alphanumeric (but keep - and _)
        tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-_]+", text.lower())
        # Filter very short tokens (single chars) but keep digits
        return [t for t in tokens if len(t) > 1]

    def _estimate_vocabulary_size(self) -> int:
        """Estimate vocabulary size for logging."""
        vocab = set()
        for tokens in self._tokenized_corpus:
            vocab.update(tokens)
        return len(vocab)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Retrieve top-k BM25-scored documents for a query.

        Args:
            query: The search query.
            top_k: Number of results (defaults to self._top_k).
            filters: Metadata filters to apply (e.g., {"paper_id": "xxx"}).

        Returns:
            List of dicts with keys: chunk_id, score, text, and metadata.

        Raises:
            RetrievalError: If the index is not built or retrieval fails.
        """
        start = time.time()
        k = top_k or self._top_k

        if self._bm25 is None:
            raise RetrievalError("BM25 index not built. Call build_index() first.")

        try:
            tokenized_query = self._tokenize(query)
            if not tokenized_query:
                return []

            scores = self._bm25.get_scores(tokenized_query)

            # Pair scores with documents and filter
            # BM25 score 0.0 means no query terms matched this document.
            # Negative scores can occur when all docs contain the query terms
            # (small corpus); those are kept for valid ranking.
            results: list[tuple[int, float]] = []
            for idx, score in enumerate(scores):
                if score == 0:
                    continue
                # Apply filters
                if filters:
                    meta = self._metadata[idx]
                    skip = False
                    for key, value in filters.items():
                        if key == "paper_id" and meta.get("paper_id") != value:
                            skip = True
                            break
                        elif key == "paper_ids" and isinstance(value, list):
                            if meta.get("paper_id") not in value:
                                skip = True
                                break
                    if skip:
                        continue
                results.append((idx, score))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:k]

            retrieved = []
            for idx, score in results:
                result = {
                    "chunk_id": self._chunk_ids[idx],
                    "score": float(score),
                    "text": self._texts[idx],
                    **{k: v for k, v in self._metadata[idx].items()
                       if k in ("paper_id", "title", "section", "subsection",
                                "page_start", "page_end")},
                }
                retrieved.append(result)

        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}", extra={"extra_data": {"query": query}})
            raise RetrievalError(f"BM25 retrieval failed: {e}") from e

        latency = time.time() - start
        logger.info(
            "BM25 retrieval complete",
            extra={"extra_data": {
                "query": query,
                "results": len(retrieved),
                "top_k": k,
                "latency_ms": round(latency * 1000, 2),
            }},
        )

        return retrieved

    def add_documents(self, documents: list[dict]) -> None:
        """Add documents to the index (rebuilds the index)."""
        all_docs = self._metadata + documents
        self.build_index(all_docs)
