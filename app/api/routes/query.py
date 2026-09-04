"""Query / QA API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_query_pipeline, get_conversation_manager
from app.pipeline.query_pipeline import QueryPipeline
from app.pipeline.conversation import ConversationManager
from app.retrieval.filters import RetrievalFilters
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("api.query")


class QueryRequest(BaseModel):
    """Request model for Q&A queries."""

    query: str = Field(..., min_length=1, max_length=2000)
    paper_ids: Optional[list[str]] = None
    filters: Optional[dict] = None
    conversation_id: Optional[str] = None
    use_standalone_query: bool = True


class QueryResponse(BaseModel):
    """Response model for Q&A queries."""

    answer: str
    citations: list[dict] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    retrieval: dict = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    insufficient_evidence: bool = False


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
    conv: ConversationManager = Depends(get_conversation_manager),
) -> QueryResponse:
    """Ask a question about the indexed papers.

    Supports conversational follow-up via the conversation_id field.
    If the query references "it", "this", or "they", the conversation
    history is used to create a standalone query.
    """
    try:
        # Handle conversational context
        if request.conversation_id and request.use_standalone_query:
            standalone, was_rewritten = conv.get_standalone_query(request.query)
            if was_rewritten:
                search_query = standalone
            else:
                search_query = request.query
        else:
            search_query = request.query

        # Build filters
        filters = None
        if request.paper_ids:
            filters = RetrievalFilters(paper_ids=request.paper_ids)

        # Run the query pipeline
        result = pipeline.answer(search_query, filters=filters)
    except Exception as e:
        logger.error(f"Query endpoint failed: {e}")
        return QueryResponse(
            answer=f"Error processing query: {e}",
            success=False,
            error=str(e),
        )

    # Store in conversation
    conv.add_turn(
        query=request.query,
        answer=result.answer,
        citations=[c.model_dump() if hasattr(c, "model_dump") else c for c in result.citations],
    )

    return QueryResponse(
        answer=result.answer,
        citations=[c.__dict__ if hasattr(c, "__dict__") else c for c in result.citations],
        evidence=[e.__dict__ if hasattr(e, "__dict__") else e for e in result.evidence],
        retrieval=result.retrieval,
        success=result.success,
        error=result.error,
        insufficient_evidence=result.insufficient_evidence,
    )


@router.post("/summarize", response_model=dict)
async def summarize(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> dict:
    """Summarize a paper.

    Requires paper_ids with exactly one paper ID.
    """
    from app.pipeline.summary_pipeline import SummaryPipeline
    from app.api.deps import get_search_service, get_llm
    from app.core.config import get_settings

    if not request.paper_ids or len(request.paper_ids) != 1:
        raise HTTPException(
            status_code=400,
            detail="Exactly one paper_id is required for summarization",
        )

    # Re-create pipeline with summary capabilities
    summary_pipeline = SummaryPipeline(
        search_service=pipeline._search_service,
        llm_provider=pipeline._llm,
    )

    result = summary_pipeline.summarize(request.paper_ids[0], request.query)
    return result


@router.post("/compare", response_model=dict)
async def compare(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> dict:
    """Compare multiple papers.

    Requires paper_ids with at least 2 paper IDs.
    """
    from app.pipeline.comparison_pipeline import ComparisonPipeline

    if not request.paper_ids or len(request.paper_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 paper_ids are required for comparison",
        )

    comparison_pipeline = ComparisonPipeline(
        search_service=pipeline._search_service,
        llm_provider=pipeline._llm,
    )

    result = comparison_pipeline.compare(request.paper_ids, request.query)
    return result


@router.post("/literature-review", response_model=dict)
async def literature_review(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> dict:
    """Generate a literature review on a topic."""
    from app.pipeline.literature_pipeline import LiteratureReviewPipeline

    review_pipeline = LiteratureReviewPipeline(
        search_service=pipeline._search_service,
        llm_provider=pipeline._llm,
    )

    result = review_pipeline.review(request.query)
    return result


@router.post("/evidence-search", response_model=dict)
async def evidence_search(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> dict:
    """Search for evidence supporting or refuting a claim."""
    from app.pipeline.evidence_pipeline import EvidencePipeline

    evidence_pipeline = EvidencePipeline(
        search_service=pipeline._search_service,
        llm_provider=pipeline._llm,
    )

    filters = None
    if request.paper_ids:
        filters = RetrievalFilters(paper_ids=request.paper_ids)

    result = evidence_pipeline.search_evidence(
        claim=request.query,
        filters=filters,
    )
    return result
