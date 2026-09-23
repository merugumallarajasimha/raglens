"""Qdrant vector database client management."""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import Settings, get_settings

logger = logging.getLogger("raglens.qdrant")


@lru_cache(maxsize=4)
def get_qdrant_client(
    host: str | None = None,
    port: int | None = None,
    timeout: float = 10.0,
) -> QdrantClient:
    """Return a cached Qdrant client for an explicit endpoint."""
    settings = get_settings()
    resolved_host = host or settings.qdrant_host
    resolved_port = settings.qdrant_port if port is None else port
    return QdrantClient(
        host=resolved_host,
        port=resolved_port,
        timeout=timeout,
    )


def get_or_create_collection(
    collection_name: str,
    embedding_dim: int,
) -> None:
    """Create a Qdrant collection if it does not already exist."""
    client = get_qdrant_client()
    collections = client.get_collections()
    names = [c.name for c in collections.collections]

    if collection_name not in names:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{collection_name}' with dim {embedding_dim}")
        except AttributeError:
            # Fallback for older client versions
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Recreated Qdrant collection '{collection_name}' with dim {embedding_dim}")


def check_qdrant_health() -> bool:
    """Check connection health to Qdrant vector database."""
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return False