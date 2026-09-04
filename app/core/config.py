"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised application settings. All values are overridable via .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Project ──────────────────────────────────────────────────────────
    project_name: str = "RAGLens"
    project_version: str = "0.1.0"
    debug: bool = False

    # ── Backend server ───────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # ── PostgreSQL ───────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "raglens"
    postgres_user: str = "raglens"
    postgres_password: str = "raglens"
    
    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Qdrant ───────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "raglens_chunks"

    # ── Ollama / LLM ─────────────────────────────────────────────────────
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_timeout: float = 120.0

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_provider: str = "sentence_transformers"
    embedding_dim: int = 384
    embedding_cache_dir: str = "data/embeddings_cache"

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_provider: str = "sentence_transformers"

    # ── Retrieval ────────────────────────────────────────────────────────
    dense_top_k: int = 20
    sparse_top_k: int = 20
    hybrid_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    # ── Chunking ────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 120
    chunking_strategy: str = "section_aware"  # fixed | paragraph | section_aware

    # ── Query processing ─────────────────────────────────────────────────
    enable_query_rewrite: bool = True
    max_retrieval_retries: int = 2
    min_rerank_score: float = 0.0

    # ── Paths ────────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    evaluation_dir: Path = Path("data/evaluation")

    def to_dict(self) -> dict:
        return self.model_dump()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance (cached)."""
    return Settings()
