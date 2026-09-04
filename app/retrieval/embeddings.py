"""Embedding provider abstraction and implementations.

Provides a configurable interface for text embedding, with support for
sentence-transformers and a mock provider for testing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from app.core.config import Settings

logger = logging.getLogger("raglens.embeddings")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

    @abstractmethod
    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts, optionally in batches."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""


class SentenceTransformersProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers (CPU-compatible)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Embedding model loaded")

    def embed_text(self, text: str) -> list[float]:
        embeddings = self._model.encode([text], convert_to_numpy=True)
        return embeddings[0].tolist()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embedding provider for testing.

    Uses hashing to generate reproducible embeddings without model loading.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic pseudo-embedding vector."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [0.0] * self._dimension
        for i in range(self._dimension):
            byte_idx = i % len(h)
            vec[i] = (h[byte_idx] / 255.0) * 2.0 - 1.0
        return vec

    def embed_text(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingCache:
    """Simple in-memory and file-based cache for embeddings.

    Prevents recomputation of embeddings for the same text.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._memory: dict[str, list[float]] = {}
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def get(self, text: str) -> Optional[list[float]]:
        """Retrieve a cached embedding, or None if not found."""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._memory:
            return self._memory[key]
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"{key}.bin")
            if os.path.exists(path):
                import struct
                with open(path, "rb") as f:
                    data = f.read()
                dim = len(data) // 4
                vec = list(struct.unpack(f"{dim}f", data))
                self._memory[key] = vec
                return vec
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in the cache."""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._memory[key] = embedding
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"{key}.bin")
            import struct
            with open(path, "wb") as f:
                f.write(struct.pack(f"{len(embedding)}f", *embedding))

    def clear(self) -> None:
        """Clear the in-memory cache."""
        self._memory.clear()


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return an embedding provider based on settings."""
    if settings is None:
        settings = Settings()

    provider_name = settings.embedding_provider

    if provider_name == "sentence_transformers":
        try:
            return SentenceTransformersProvider(settings.embedding_model)
        except Exception as e:
            logger.warning(
                f"Failed to load embedding model {settings.embedding_model}: {e}. "
                "Falling back to mock provider."
            )
            return MockEmbeddingProvider(dimension=settings.embedding_dim)
    elif provider_name == "mock":
        return MockEmbeddingProvider(dimension=settings.embedding_dim)
    else:
        return MockEmbeddingProvider(dimension=settings.embedding_dim)


@lru_cache(maxsize=1)
def get_embedding_cache(settings: Settings | None = None) -> EmbeddingCache:
    """Return a cached embedding cache."""
    if settings is None:
        settings = Settings()
    return EmbeddingCache(cache_dir=settings.embedding_cache_dir)
