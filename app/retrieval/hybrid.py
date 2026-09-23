"""Hybrid retrieval — combines dense and sparse retrieval via RRF.

Uses Reciprocal Rank Fusion (RRF) to merge ranking lists from dense
(Qdrant vector search) and sparse (BM25) retrievers into a unified
candidate set.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings
from app.core.exceptions import RetrievalError
from app.database.vector_store import RetrievedChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever

logger = logging.getLogger("raglens.retrieval.hybrid")


@dataclass
class HybridResult:
    """A result from hybrid retrieval with scores from both retrievers."""

    chunk_id: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rank: Optional[int] = None
    text: str = ""
    paper_id: str = ""
    title: str = ""
    section: Optional[str] = None
    subsection: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: int = 0
    metadata: Optional[dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridRetriever:
    """Combines dense and sparse retrieval using Reciprocal Rank Fusion.

    Args:
        dense_retriever: The dense retriever instance.
        sparse_retriever: The sparse retriever instance.
        dense_weight: Weight for dense retriever scores (default 0.5).
        sparse_weight: Weight for sparse retriever scores (default 0.5).
        rrf_k: RRF constant k (default 60). Controls score decay.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[HybridResult]:
        """Retrieve hybrid top-k results for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.
            filters: Metadata filters to pass to both retrievers.

        Returns:
            List of HybridResult objects sorted by RRF score descending.
        """
        start = time.time()

        # Sanitize filters: convert empty dicts/falsy values to None
        active_filters = filters if filters else None

        # Retrieve from both retrievers with larger candidate pool for fusion
        # Fetch at least 30, or 3x top_k, from each retriever
        dense_k = max(top_k * 3, 30)
        sparse_k = max(top_k * 3, 30)

        dense_results = []
        sparse_results = []

        try:
            dense_results = self._dense.retrieve(query, top_k=dense_k, filters=active_filters)
        except Exception as e:
            logger.warning(f"Dense retrieval failed during hybrid search: {e}")

        try:
            sparse_results = self._sparse.retrieve(query, top_k=sparse_k, filters=active_filters)
        except Exception as e:
            logger.warning(f"Sparse retrieval failed during hybrid search: {e}")

        # Build RRF fused ranking
        rrf_scores: dict[str, dict] = {}

        # Dense results (higher score = more relevant)
        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk_id
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = self._init_result(result)
            rrf_scores[chunk_id]["dense_score"] = result.score
            rrf_scores[chunk_id]["dense_rank"] = rank + 1
            rrf_scores[chunk_id]["rrf_score"] += (
                self._dense_weight * self._rrf(rank + 1)
            )

        # Sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result.get("chunk_id", str(rank))
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = self._init_result_from_sparse(result)
            rrf_scores[chunk_id]["sparse_score"] = float(result.get("score", 0.0))
            rrf_scores[chunk_id]["sparse_rank"] = rank + 1
            rrf_scores[chunk_id]["rrf_score"] += (
                self._sparse_weight * self._rrf(rank + 1)
            )

        # Sort by RRF score and take top_k
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid]["rrf_score"],
            reverse=True,
        )

        results = []
        for i, chunk_id in enumerate(sorted_ids[:top_k]):
            data = rrf_scores[chunk_id]
            hr = HybridResult(
                chunk_id=chunk_id,
                score=data["rrf_score"],
                dense_score=data.get("dense_score"),
                sparse_score=data.get("sparse_score"),
                text=data.get("text", ""),
                paper_id=data.get("paper_id", ""),
                title=data.get("title", ""),
                section=data.get("section"),
                subsection=data.get("subsection"),
                page_start=data.get("page_start"),
                page_end=data.get("page_end"),
                token_count=data.get("token_count", 0),
                metadata=data.get("metadata", {}),
            )
            hr.rank = i + 1
            results.append(hr)

        # Deduplicate by text to avoid identical chunks from dense+sparse fusion
        results = self._deduplicate_chunks(results)

        latency = time.time() - start
        logger.info(
            "Hybrid retrieval complete",
            extra={"extra_data": {
                "query": query,
                "dense_results": len(dense_results),
                "sparse_results": len(sparse_results),
                "fused_results": len(results),
                "top_k": top_k,
                "latency_ms": round(latency * 1000, 2),
            }},
        )

        return results

    def _rrf(self, rank: int) -> float:
        """Reciprocal Rank Fusion score for a given rank.

        score = 1 / (rank + k)
        """
        return 1.0 / (rank + self._rrf_k)

    def _init_result(self, result: RetrievedChunk) -> dict:
        """Initialize a result dict from a RetrievedChunk."""
        return {
            "rrf_score": 0.0,
            "dense_score": None,
            "sparse_score": None,
            "text": result.text,
            "paper_id": result.paper_id,
            "title": result.title,
            "section": result.section,
            "subsection": result.subsection,
            "page_start": result.page_start,
            "page_end": result.page_end,
            "token_count": result.token_count,
            "metadata": result.metadata,
        }

    def _init_result_from_sparse(self, result: dict) -> dict:
        """Initialize a result dict from a sparse retrieval result."""
        return {
            "rrf_score": 0.0,
            "dense_score": None,
            "sparse_score": None,
            "text": result.get("text", ""),
            "paper_id": result.get("paper_id", ""),
            "title": result.get("title", ""),
            "section": result.get("section"),
            "subsection": result.get("subsection"),
            "page_start": result.get("page_start"),
            "page_end": result.get("page_end"),
            "token_count": result.get("token_count", 0),
            "metadata": result.get("metadata", {}),
        }

    def _deduplicate_chunks(self, chunks: list[HybridResult]) -> list[HybridResult]:
        """Deduplicate chunks by text content before returning to reranker.

        Hybrid search (dense + sparse) can return the same chunk from both
        retrievers. This deduplicates by exact text match.
        """
        seen_texts = set()
        unique_chunks = []
        for chunk in chunks:
            text = chunk.text.strip() if chunk.text else ""
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_chunks.append(chunk)
        return unique_chunks