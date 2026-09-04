# RAGLens Architecture

## Overview

RAGLens is a citation-aware Research Paper Retrieval-Augmented Generation (RAG) system.
It allows users to search and ask questions about a corpus of research papers and
receive grounded answers with citations, section/page references, and supporting evidence.

## Architecture Diagram

```
User
  ↓
Frontend (Streamlit)
  ↓ HTTP/JSON
FastAPI Backend
  │
  ├─ Query Understanding
  │    ├─ Query Classification
  │    └─ Query Rewriting
  │
  ├─ Hybrid Retrieval
  │    ├─ Dense Retrieval (Qdrant + embeddings)
  │    ├─ Sparse Retrieval (BM25)
  │    └─ Result Fusion (RRF)
  │
  ├─ Reranking (cross-encoder)
  │
  ├─ Evidence Filtering
  │
  ├─ Context Builder (structured LLM context)
  │
  ├─ LLM Generation (Ollama)
  │
  ├─ Citation Engine (backend-generated citations)
  │
  └─ Citation Validation (fabricated citation detection)
       │
       ↓
  Answer + Sources + Evidence + Retrieval Transparency
```

## Components

### Data Layer
- **PostgreSQL**: Structured metadata (papers, authors, chunks, sections)
- **Qdrant**: Vector embeddings for dense retrieval
- **Filesystem**: Raw PDFs and processed data

### Ingestion Pipeline
1. **PDF Parser** (`pdf_parser.py`): Extracts text, structure, page boundaries
2. **Metadata Extractor** (`metadata_extractor.py`): Title, authors, abstract, year, source
3. **Structure Parser** (`structure_parser.py`): Section/subsection hierarchy
4. **Chunker** (`chunker.py`): Structure-aware chunking with overlap
5. **Pipeline** (`pipeline.py`): Orchestrates the ingestion flow

### Retrieval Layer
1. **Dense Retriever** (`dense.py`): Embeds query, searches Qdrant
2. **Sparse Retriever** (`sparse.py`): BM25 lexical search
3. **Hybrid Retriever** (`hybrid.py`): RRF fusion of dense + sparse
4. **Reranker** (`reranker.py`): Cross-encoder relevance scoring
5. **Filters** (`filters.py`): Metadata-based filtering (paper_id, year, source, etc.)

### Generation Layer
1. **LLM Abstraction** (`llm.py`): Pluggable LLM providers (Ollama, future API models)
2. **Prompts** (`prompts.py`): Strict grounded-generation prompts
3. **Context Builder** (`context_builder.py`): Structured context from reranked chunks
4. **Citation Engine** (`citation.py`): Backend-generated stable citation IDs

### Pipeline Layer
- **Query Pipeline** (`query_pipeline.py`): Normal Q&A
- **Comparison Pipeline** (`comparison_pipeline.py`): Multi-paper comparison
- **Summary Pipeline** (`summary_pipeline.py`): Paper summarization
- **Literature Pipeline** (`literature_pipeline.py`): Literature review mode

### Evaluation
- **Retrieval Metrics** (`retrieval_metrics.py`): Recall@K, MRR, NDCG
- **Generation Metrics** (`generation_metrics.py`): Ragas + custom citation metrics
- **Experiments** (`experiments.py`): Ablation study runner

## Design Decisions

### Why Hybrid Retrieval?
Dense retrieval captures semantic similarity but may miss exact terminology matches.
Sparse retrieval (BM25) excels at exact term matching but lacks semantic breadth.
Hybrid retrieval combines both via Reciprocal Rank Fusion to maximize recall coverage.

### Why Reranking?
Initial retrieval produces candidates that are not optimally ordered. A cross-encoder
reranker considers query-candidate interaction to produce precise relevance scores.

### Why Backend-Generated Citations?
Relying on the LLM to track citation IDs leads to hallucinated or fabricated references.
The backend knows the ground-truth mapping (chunk → paper → section → page), so citation
IDs are assigned deterministically before generation and validated afterward.

## Scalability

The architecture is designed to scale from ~100 papers to 5000+:
- Incremental ingestion with deduplication
- Embedding caching (no recomputation)
- Batch processing for efficiency
- Configurable batch sizes and concurrency
- CPU-compatible model choices for hardware-constrained environments
