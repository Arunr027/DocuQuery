from app.retrieval.hybrid_search import search_hybrid_chunks
from fastapi import APIRouter, HTTPException
from qdrant_client.http.exceptions import UnexpectedResponse
from app.retrieval.keyword_search import search_keyword_chunks

from app.embeddings.embedder import embed_query
from app.retrieval.vector_store import search_chunks
from app.schemas.query import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)

router = APIRouter(prefix="/query", tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    question = request.question.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Question must contain at least 3 non-space characters.",
        )

    try:
        query_vector = embed_query(question)

        results = search_chunks(
            query_vector=query_vector,
            limit=request.top_k,
        )

        return SearchResponse(
            question=question,
            results=[
                SearchResult(**result)
                for result in results
            ],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Semantic search failed. Check the API logs.",
        ) from exc

@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(request: SearchRequest):
    question = request.question.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Question must contain at least 3 non-space characters.",
        )

    try:
        results = search_keyword_chunks(
            question=question,
            limit=request.top_k,
        )

        return SearchResponse(
            question=question,
            results=[
                SearchResult(**result)
                for result in results
            ],
        )

    except Exception as exc:
     print("KEYWORD SEARCH ERROR:", repr(exc))

    raise HTTPException(
        status_code=503,
        detail="Keyword search failed. Check the API logs.",
    ) from exc
@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    question = request.question.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Question must contain at least 3 non-space characters.",
        )

    try:
        results = search_hybrid_chunks(
            question=question,
            limit=request.top_k,
        )

        return SearchResponse(
            question=question,
            results=[
                SearchResult(**result)
                for result in results
            ],
        )

    except Exception as exc:
        print("HYBRID SEARCH ERROR:", repr(exc))

        raise HTTPException(
            status_code=503,
            detail="Hybrid search failed. Check the API logs.",
        ) from exc
