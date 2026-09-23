"""Search service — orchestrates the complete retrieval pipeline.

Manages: corpus building, dense retrieval, BM25 indexing, hybrid fusion,
and reranking. Acts as the main entry point for all search operations.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.database.vector_store import QdrantVectorStore, RetrievedChunk
from app.ingestion.chunker import Chunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.hybrid import HybridRetriever, HybridResult
from app.retrieval.reranker import Reranker
from app.retrieval.filters import RetrievalFilters

logger = logging.getLogger("raglens.services.search")


class SearchService:
    """High-level search service orchestrating the full retrieval pipeline.

    Manages the lifecycle of the dense retriever, sparse (BM25) retriever,
    hybrid merger, and reranker.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_provider: Optional[object] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if settings is None:
            settings = get_settings()

        self._settings = settings
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

        # Lazy-initialized components
        self._dense_retriever: Optional[DenseRetriever] = None
        self._sparse_retriever: Optional[SparseRetriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[Reranker] = None

    def get_dense_retriever(self) -> DenseRetriever:
        """Lazy-initialize and return the dense retriever."""
        if self._dense_retriever is None:
            if self._embedding_provider is None:
                from app.retrieval.embeddings import get_embedding_provider
                self._embedding_provider = get_embedding_provider(self._settings)

            self._dense_retriever = DenseRetriever(
                vector_store=self._vector_store,
                embedding_provider=self._embedding_provider,
            )
        return self._dense_retriever

    def get_sparse_retriever(self) -> SparseRetriever:
        """Lazy-initialize and return the sparse retriever."""
        if self._sparse_retriever is None:
            self._sparse_retriever = SparseRetriever(
                top_k=self._settings.sparse_top_k,
            )
        return self._sparse_retriever

    def get_hybrid_retriever(self) -> HybridRetriever:
        """Lazy-initialize and return the hybrid retriever."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                dense_retriever=self.get_dense_retriever(),
                sparse_retriever=self.get_sparse_retriever(),
                dense_weight=self._settings.dense_weight,
                sparse_weight=self._settings.sparse_weight,
                rrf_k=self._settings.rrf_k,
            )
        return self._hybrid_retriever

    def get_reranker(self) -> Reranker:
        """Lazy-initialize and return the reranker."""
        if self._reranker is None:
            self._reranker = Reranker.from_settings(self._settings)
        return self._reranker

    def build_bm25_index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of chunks.

        Call this after ingesting papers to enable sparse retrieval.
        """
        corpus = []
        for chunk in chunks:
            corpus.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "paper_id": chunk.paper_id,
                "title": chunk.title,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            })

        self.get_sparse_retriever().build_index(corpus)
        logger.info(
            "BM25 index rebuilt",
            extra={"extra_data": {"documents": len(corpus)}},
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None,
        rerank: bool = True,
    ) -> list[HybridResult]:
        """Execute a hybrid search with optional reranking.

        Args:
            query: The search query.
            top_k: Number of final results (defaults to rerank_top_k if reranking,
                   otherwise hybrid_top_k).
            filters: Metadata filters.
            rerank: Whether to apply reranking.

        Returns:
            List of HybridResult objects.
        """
        start = time.time()

        if top_k is None:
            top_k = self._settings.rerank_top_k if rerank else self._settings.hybrid_top_k

        # Convert filters to dict for retrievers
        filter_dict = self._filters_to_dict(filters)

        # Step 1: Dense + Sparse fusion (RRF)
        # Fetch a larger candidate pool (at least 30, or 3x top_k) for better fusion
        candidate_pool_k = max(top_k * 3, 30)
        hybrid = self.get_hybrid_retriever()
        results = hybrid.retrieve(
            query=query,
            top_k=candidate_pool_k,
            filters=filter_dict,
        )

        # Step 2: Rerank
        if rerank and results:
            reranker = self.get_reranker()
            results = reranker.rerank(
                query=query,
                candidates=results,
                top_k=top_k,
            )
        else:
            results = results[:top_k]

        # Step 3: Apply post-filtering
        if filters:
            from app.retrieval.filters import apply_filters
            results = apply_filters(results, filters)

        latency = time.time() - start
        logger.info(
            "Search complete",
            extra={"extra_data": {
                "query": query,
                "results": len(results),
                "top_k": top_k,
                "rerank": rerank,
                "latency_ms": round(latency * 1000, 2),
            }},
        )

        return results

    def _filters_to_dict(self, filters: Optional[RetrievalFilters]) -> Optional[dict]:
        """Convert RetrievalFilters to a dict for retrievers."""
        if filters is None:
            return None

        result: dict = {}
        if filters.paper_ids and len(filters.paper_ids) == 1:
            result["paper_id"] = filters.paper_ids[0]
        elif filters.paper_ids:
            result["paper_ids"] = filters.paper_ids

        return result if result else None
