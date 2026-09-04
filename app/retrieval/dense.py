"""Dense retrieval — query embedding → Qdrant vector search."""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.core.config import Settings
from app.core.exceptions import RetrievalError
from app.database.vector_store import QdrantVectorStore, RetrievedChunk
from app.retrieval.embeddings import EmbeddingProvider

logger = logging.getLogger("raglens.retrieval.dense")


class DenseRetriever:
    """Retrieves documents using dense vector similarity search.

    Args:
        vector_store: The Qdrant vector store.
        embedding_provider: The embedding provider for query encoding.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k similar chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.
            filters: Metadata filters (paper_id, year, source, etc.).

        Returns:
            List of RetrievedChunk objects sorted by score descending.

        Raises:
            RetrievalError: If retrieval fails.
        """
        start = time.time()

        try:
            embedding = self._embedding_provider.embed_text(query)
            results = self._vector_store.search(
                query_vector=embedding,
                top_k=top_k,
                score_threshold=score_threshold,
                filter_conditions=filters,
            )
        except Exception as e:
            logger.error(
                f"Dense retrieval failed: {e}",
                extra={"extra_data": {"query": query}},
            )
            raise RetrievalError(f"Dense retrieval failed: {e}") from e

        latency = time.time() - start
        logger.info(
            "Dense retrieval complete",
            extra={"extra_data": {
                "query": query,
                "results": len(results),
                "top_k": top_k,
                "latency_ms": round(latency * 1000, 2),
            }},
        )

        return results
