"""In-memory vector store for testing — implements the same interface as QdrantVectorStore."""

from __future__ import annotations

import math
import logging
from typing import Optional

from app.database.vector_store import RetrievedChunk
from app.ingestion.chunker import Chunk

logger = logging.getLogger("raglens.test.vector_store")


class InMemoryVectorStore:
    """Simple in-memory vector store for testing.

    Stores vectors and their payloads, supports similarity search via
    cosine similarity. Not for production use.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self._embedding_dim = embedding_dim
        self._vectors: dict[str, list[float]] = {}
        self._payloads: dict[str, dict] = {}

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings")

        for chunk, emb in zip(chunks, embeddings):
            self._vectors[chunk.chunk_id] = emb
            self._payloads[chunk.chunk_id] = {
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
            }

        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        import math

        results = []
        for chunk_id, vector in self._vectors.items():
            payload = self._payloads[chunk_id]

            # Apply filters
            if filter_conditions:
                matches = True
                for key, value in filter_conditions.items():
                    if key == "paper_id" and payload.get("paper_id") != value:
                        matches = False
                        break
                    elif key == "paper_ids" and payload.get("paper_id") not in value:
                        matches = False
                        break
                if not matches:
                    continue

            # Compute cosine similarity
            score = self._cosine_similarity(query_vector, vector)

            if score_threshold is not None and score < score_threshold:
                continue

            results.append(RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                score=score,
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

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def count(self) -> int:
        return len(self._vectors)

    def delete_by_paper_id(self, paper_id: str) -> int:
        ids_to_delete = [
            cid for cid, payload in self._payloads.items()
            if payload.get("paper_id") == paper_id
        ]
        for cid in ids_to_delete:
            del self._vectors[cid]
            del self._payloads[cid]
        return len(ids_to_delete)
