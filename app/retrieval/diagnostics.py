"""Retrieval diagnostics — expose raw dense/sparse scores and Qdrant scroll checks.

Useful for isolating whether a missing chunk is a Recall issue (never entered
the candidate pool) or a Ranking issue (entered but got outranked).

Usage:
    from app.retrieval.diagnostics import debug_retrieval, scroll_for_text
    debug_retrieval("how many layers does the transformer contain?")
    scroll_for_text("identical layers")
"""

from __future__ import annotations

import logging
from typing import Optional

from app.database.vector_store import QdrantVectorStore
from app.retrieval.embeddings import EmbeddingProvider, get_embedding_provider
from app.retrieval.sparse import SparseRetriever

logger = logging.getLogger("raglens.retrieval.diagnostics")


def debug_retrieval(
    query: str,
    top_k: int = 20,
    vector_store: Optional[QdrantVectorStore] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
) -> dict:
    """Print raw dense + sparse scores for a query before reranking.

    Returns a dict with the dense and sparse result lists so callers can
    inspect them programmatically.
    """
    if vector_store is None:
        from app.database.vector_store import QdrantVectorStore
        from app.core.config import get_settings
        vector_store = QdrantVectorStore.from_settings(get_settings())
    if embedding_provider is None:
        embedding_provider = get_embedding_provider()

    print("\n" + "=" * 70)
    print(f"RAW RETRIEVAL DIAGNOSTICS — query: {query!r}")
    print("=" * 70)

    # ── 1. Raw Dense Vector Search ──
    embedding = embedding_provider.embed_text(query)
    dense_results = vector_store.search(
        query_vector=embedding,
        top_k=top_k,
        score_threshold=None,
        filter_conditions=None,
    )

    print("\n--- RAW DENSE SEARCH RESULTS ---")
    dense_out = []
    for i, res in enumerate(dense_results):
        section = res.section or "N/A"
        text_snippet = (res.text or "")[:120].replace("\n", " ")
        print(
            f"[{i + 1}] Score: {res.score:.4f} | "
            f"Section: {section} | Text: {text_snippet}"
        )
        dense_out.append({
            "rank": i + 1,
            "score": res.score,
            "section": section,
            "chunk_id": res.chunk_id,
            "paper_id": res.paper_id,
            "text": res.text,
        })

    # ── 2. Raw Sparse / BM25 Search ──
    sparse_results = []
    sparse_out = []
    if sparse_retriever is not None and sparse_retriever._bm25 is not None:
        sparse_raw = sparse_retriever.retrieve(query, top_k=top_k)
        print("\n--- RAW SPARSE (BM25) SEARCH RESULTS ---")
        for i, res in enumerate(sparse_raw):
            section = res.get("section") or "N/A"
            text_snippet = (res.get("text") or "")[:120].replace("\n", " ")
            print(
                f"[{i + 1}] Score: {res.get('score', 0):.4f} | "
                f"Section: {section} | Text: {text_snippet}"
            )
            sparse_out.append({
                "rank": i + 1,
                "score": res.get("score", 0),
                "section": section,
                "chunk_id": res.get("chunk_id"),
                "paper_id": res.get("paper_id"),
                "text": res.get("text"),
            })
        sparse_results = sparse_raw
    else:
        print("\n--- SPARSE (BM25) SEARCH ---")
        print("(BM25 index not available — sparse_retriever is None or not built)")

    return {
        "query": query,
        "dense_results": dense_out,
        "sparse_results": sparse_out,
    }


def scroll_for_text(
    text_fragment: str,
    collection_name: Optional[str] = None,
    limit: int = 10,
    vector_store: Optional[QdrantVectorStore] = None,
) -> list[dict]:
    """Scroll Qdrant for chunks whose payload text contains a substring.

    Use this to confirm whether a chunk exists in the collection at all,
    independent of vector similarity ranking.

    Returns a list of dicts with chunk_id, section, and text.
    """
    if vector_store is None:
        from app.core.config import get_settings
        vector_store = QdrantVectorStore.from_settings(get_settings())

    if collection_name is None:
        collection_name = vector_store._collection_name

    from qdrant_client.http import models as qmodels

    print("\n" + "=" * 70)
    print(f"QDRANT SCROLL — collection: {collection_name!r}, fragment: {text_fragment!r}")
    print("=" * 70)

    scroll_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="text",
                match=qmodels.MatchText(text=text_fragment),
            )
        ]
    )

    try:
        scroll_result, _ = vector_store._client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )
    except Exception as e:
        print(f"Scroll failed (MatchText may be unsupported): {e}")
        # Fallback: fetch all points and filter client-side
        return _scroll_fallback(vector_store, text_fragment, limit)

    print(f"Found {len(scroll_result)} chunks containing {text_fragment!r}:")
    results = []
    for point in scroll_result:
        payload = point.payload or {}
        section = payload.get("section") or "N/A"
        text = payload.get("text") or ""
        print(f"\n  ID: {point.id}")
        print(f"  Section: {section}")
        print(f"  Text: {text[:200]}...")
        results.append({
            "chunk_id": point.id,
            "section": section,
            "text": text,
        })

    if not scroll_result:
        print("  (no matches)")

    return results


def _scroll_fallback(
    vector_store: QdrantVectorStore,
    text_fragment: str,
    limit: int,
) -> list[dict]:
    """Fallback scroll that fetches points and filters client-side."""
    from qdrant_client.http import models as qmodels

    results: list[dict] = []
    offset = None
    while True:
        batch, offset = vector_store._client.scroll(
            collection_name=vector_store._collection_name,
            scroll_filter=None,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for point in batch:
            text = (point.payload or {}).get("text") or ""
            if text_fragment.lower() in text.lower():
                results.append({
                    "chunk_id": point.id,
                    "section": (point.payload or {}).get("section") or "N/A",
                    "text": text,
                })
                if len(results) >= limit:
                    return results
        if offset is None:
            break

    print(f"  (fallback found {len(results)} matches)")
    return results


def collection_point_count(
    collection_name: Optional[str] = None,
    vector_store: Optional[QdrantVectorStore] = None,
) -> int:
    """Return the point count for the configured collection."""
    if vector_store is None:
        from app.core.config import get_settings
        vector_store = QdrantVectorStore.from_settings(get_settings())
    return vector_store.count()