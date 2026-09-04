# RAGLens API Documentation

RAGLens is a citation-aware Research Paper Retrieval-Augmented Generation (RAG) system.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Start infrastructure (PostgreSQL + Qdrant)

```bash
docker-compose up -d
```

### 4. Run the backend

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run the frontend (optional)

```bash
streamlit run frontend/streamlit_app.py
```

### 6. Ingest papers

```bash
python scripts/ingest.py data/raw/*.pdf
```

## API Endpoints

### Health

- `GET /` — Application info
- `GET /health` — Health check with dependency status
- `GET /health/liveness` — Liveness probe

### Papers

- `POST /papers/upload` — Upload a PDF file
- `POST /papers/ingest` — Ingest a paper from a directory
- `GET /papers` — List all papers
- `GET /papers/{paper_id}` — Get paper metadata
- `GET /papers/{paper_id}/chunks` — Get chunks for a paper

### Search & Query

- `POST /search` — Search the corpus
- `POST /query` — Ask a question
- `POST /summarize` — Summarize a paper
- `POST /compare` — Compare multiple papers
- `POST /literature-review` — Generate a literature review

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full architecture.
