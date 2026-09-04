"""Qdrant vector store for dense retrieval.

Provides a typed interface for storing and searching vector embeddings
with metadata payload (chunk_id, paper_id, section, page, etc.).
"""

from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import Settings
from app.core.exceptions import VectorStoreError
from app.ingestion.chunker import Chunk

logger = logging.getLogger("raglens.vector_store")


class RetrievedChunk:
    """A chunk retrieved from the vector store with a similarity score."""

    def __init__(
        self,
        chunk_id: str,
        score: float,
        text: str,
        paper_id: str,
        title: str,
        section: Optional[str] = None,
        subsection: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        token_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.text = text
        self.paper_id = paper_id
        self.title = title
        self.section = section
        self.subsection = subsection
        self.page_start = page_start
        self.page_end = page_end
        self.token_count = token_count
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "paper_id": self.paper_id,
            "title": self.title,
            "section": self.section,
            "subsection": self.subsection,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


class QdrantVectorStore:
    """Manages Qdrant collections for dense retrieval."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "raglens_chunks",
        embedding_dim: int = 384,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "QdrantVectorStore":
        """Create a vector store configured from application settings."""
        if settings is None:
            settings = Settings()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=10.0,
        )
        return cls(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding_dim=settings.embedding_dim,
        )

    def create_collection(self, force: bool = False) -> None:
        """Create the collection if it doesn't exist."""
        collections = self._client.get_collections()
        names = [c.name for c in collections.collections]

        if self._collection_name in names:
            if force:
                self._client.delete_collection(self._collection_name)
            else:
                logger.info(f"Collection {self._collection_name} already exists")
                return

        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=qmodels.VectorParams(
                size=self._embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            "Collection created",
            extra={"extra_data": {
                "collection": self._collection_name,
                "dim": self._embedding_dim,
            }},
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert chunks with their embeddings into the collection.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of embedding vectors (same order as chunks).

        Returns:
            Number of points upserted.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        if not chunks:
            return 0

        points = []
        for chunk, emb in zip(chunks, embeddings):
            points.append(qmodels.PointStruct(
                id=chunk.chunk_id,
                vector=emb,
                payload={
                    "paper_id": chunk.paper_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "metadata": chunk.metadata,
                },
            ))

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

        logger.info(
            "Chunks upserted to Qdrant",
            extra={"extra_data": {
                "collection": self._collection_name,
                "count": len(points),
            }},
        )
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Search for similar vectors in the collection.

        Args:
            query_vector: The query embedding.
            top_k: Number of results to return.
            score_threshold: Minimum score for results.
            filter_conditions: Optional payload filters (e.g., {"paper_id": "xxx"}).

        Returns:
            List of RetrievedChunk objects sorted by score descending.
        """
        search_filter = self._build_filter(filter_conditions)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            filter=search_filter,
            with_payload=True,
        )

        retrieved: list[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            retrieved.append(RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                score=hit.score,
                text=payload.get("text", ""),
                paper_id=payload.get("paper_id", ""),
                title=payload.get("title", ""),
                section=payload.get("section"),
                subsection=payload.get("subsection"),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                token_count=payload.get("token_count", 0),
                metadata=payload.get("metadata", {}),
            ))

        return retrieved

    def delete_by_paper_id(self, paper_id: str) -> int:
        """Delete all points for a paper."""
        result = self._client.delete(
            collection_name=self._collection_name,
            points_selector=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="paper_id",
                    match=qmodels.MatchValue(paper_id),
                )],
            ),
            wait=True,
        )
        return result

    def count(self) -> int:
        """Return the total number of points in the collection."""
        result = self._client.count(self._collection_name)
        return result.count

    def collection_info(self) -> dict:
        """Return collection statistics."""
        info = self._client.get_collection(self._collection_name)
        return {
            "name": self._collection_name,
            "points_count": info.points_count,
            "embedding_dim": self._embedding_dim,
        }

    def _build_filter(self, conditions: Optional[dict]) -> Optional[qmodels.Filter]:
        """Build a Qdrant filter from conditions dict."""
        if not conditions:
            return None

        must_conditions = []
        for key, value in conditions.items():
            must_conditions.append(qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchValue(value=str(value)) if not isinstance(value, list)
                else qmodels.MatchAny(any=[str(v) for v in value]),
            ))

        if must_conditions:
            return qmodels.Filter(must=must_conditions)
        return None
