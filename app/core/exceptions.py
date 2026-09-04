"""Custom exception hierarchy for RAGLens."""

from __future__ import annotations

from typing import Any


class RAGLensError(Exception):
    """Base exception for all RAGLens errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class ConfigError(RAGLensError):
    """Raised when configuration is invalid."""


class IngestionError(RAGLensError):
    """Raised when document ingestion fails."""


class PDFParseError(IngestionError):
    """Raised when a PDF cannot be parsed."""


class EmbeddingError(RAGLensError):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGLensError):
    """Raised when vector store operations fail."""


class RetrievalError(RAGLensError):
    """Raised when retrieval fails."""


class RerankError(RAGLensError):
    """Raised when reranking fails."""


class LLMError(RAGLensError):
    """Raised when LLM generation fails."""


class CitationError(RAGLensError):
    """Raised when citation validation fails."""


class DatabaseError(RAGLensError):
    """Raised when database operations fail."""


class EvaluationError(RAGLensError):
    """Raised when evaluation fails."""
