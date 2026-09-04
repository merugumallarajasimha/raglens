"""Qdrant vector database client management."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import get_settings

_client: Optional[QdrantClient] = None


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client instance."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=5.0,
        )
    return _client


def get_or_create_collection(
    collection_name: str,
    embedding_dim: int,
) -> None:
    """Create a Qdrant collection if it does not already exist."""
    client = get_qdrant_client()
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    if collection_name not in names:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
