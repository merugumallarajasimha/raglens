"""FastAPI dependency injection for services.

Provides shared instances of the search service, LLM provider,
and query pipeline. Handles graceful degradation when
external services (Qdrant, PostgreSQL, Ollama) are unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError, VectorStoreError
from app.database.qdrant import get_qdrant_client
from app.database.vector_store import QdrantVectorStore
from app.generation.llm import get_llm_provider, LLMProvider
from app.retrieval.embeddings import (
    get_embedding_provider as _create_embedding_provider,
    EmbeddingProvider,
)
from app.services.search_service import SearchService
from app.pipeline.query_pipeline import QueryPipeline
from app.pipeline.comparison_pipeline import ComparisonPipeline
from app.pipeline.summary_pipeline import SummaryPipeline
from app.pipeline.literature_pipeline import LiteratureReviewPipeline
from app.pipeline.evidence_pipeline import EvidencePipeline
from app.pipeline.conversation import ConversationManager

logger = logging.getLogger("raglens.api.deps")

# Shared singleton instances
_search_service: Optional[SearchService] = None
_llm_provider: Optional[LLMProvider] = None
_embedding_provider: Optional[EmbeddingProvider] = None
_query_pipeline: Optional[QueryPipeline] = None
_conversation_manager: Optional[ConversationManager] = None


def get_vector_store(settings: Settings = Depends(get_settings)) -> QdrantVectorStore:
    """Provide a Qdrant vector store."""
    try:
        return QdrantVectorStore.from_settings(settings)
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise HTTPException(
            status_code=503,
            detail="Vector store (Qdrant) is not available",
        )


def get_embedding_provider(settings: Settings = Depends(get_settings)) -> EmbeddingProvider:
    """Provide an embedding provider (singleton)."""
    global _embedding_provider
    if _embedding_provider is None:
        try:
            _embedding_provider = _create_embedding_provider(settings)
        except Exception as e:
            logger.warning(f"Failed to create embedding provider: {e}. Using mock.")
            from app.retrieval.embeddings import MockEmbeddingProvider
            _embedding_provider = MockEmbeddingProvider(dimension=settings.embedding_dim)
    return _embedding_provider


def get_llm(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Provide an LLM provider (singleton)."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider(settings)
    return _llm_provider


def get_search_service(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    settings: Settings = Depends(get_settings),
) -> SearchService:
    """Provide a shared search service."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            settings=settings,
        )
    return _search_service


def get_query_pipeline(
    search_service: SearchService = Depends(get_search_service),
    llm: LLMProvider = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> QueryPipeline:
    """Provide a query pipeline."""
    global _query_pipeline
    if _query_pipeline is None:
        _query_pipeline = QueryPipeline(
            search_service=search_service,
            llm_provider=llm,
            settings=settings,
        )
    return _query_pipeline


def get_conversation_manager() -> ConversationManager:
    """Provide a conversation manager (singleton)."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager(max_history=10)
    return _conversation_manager
