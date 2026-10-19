
import logging

from fastapi import APIRouter, HTTPException

from app.retrieval.hybrid_search import search_hybrid_chunks
from app.schemas.answer import (
    AnswerRequest,
    AnswerResponse,
    SourceCitation,
)
from app.services.gemini_answer import generate_grounded_answer


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["RAG Answers"],
)


def answer_indicates_insufficient_evidence(answer: str) -> bool:
    """Detect common statements that the documents lack an answer."""
    normalized = " ".join(answer.lower().split())

    refusal_phrases = (
        "based on the provided context, the documents do not provide information",
        "based on the provided context, the documents do not contain information",
        "the provided documents do not contain information",
        "the provided documents do not contain enough information",
        "the documents do not contain information",
        "the documents do not provide enough information",
        "the documents do not contain enough information",
        "i could not find relevant information in the uploaded documents",
        "i cannot answer based on the provided documents",
        "i can't answer based on the provided documents",
        "the context does not provide enough information",
        "the context does not contain enough information",
        "based on the provided documents, there is no information",
        "the provided documents do not provide information",
        "there is no information in the provided documents",
        "the supplied documents do not provide enough information",
        "the supplied documents do not provide information",
        "the supplied documents do not contain information",
        "the supplied documents do not contain enough information",
    )

    return any(phrase in normalized for phrase in refusal_phrases)


@router.post("/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest):
    question = request.question.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Question must contain at least 3 non-space characters.",
        )

    try:
        # Retrieve chunks using hybrid search, RRF,
        # reranking, and Redis caching.
        chunks = search_hybrid_chunks(
            question=question,
            limit=request.top_k,
        )

        if not chunks:
            return AnswerResponse(
                question=question,
                answer=(
                    "I could not find relevant information "
                    "in the uploaded documents to answer this question."
                ),
                sources=[],
            )

        # Generate an answer grounded in the retrieved chunks.
        answer = await generate_grounded_answer(
            question=question,
            chunks=chunks,
        )

        # Avoid citations when the documents cannot answer.
        if answer_indicates_insufficient_evidence(answer):
            return AnswerResponse(
                question=question,
                answer=answer,
                sources=[],
            )

        # Attach source metadata to the answer.
        sources = [
            SourceCitation(
                source_id=f"S{index}",
                filename=chunk.get("filename"),
                page_number=chunk.get("page_number"),
                chunk_index=chunk.get("chunk_index"),
                score=chunk.get("score"),
                text=chunk.get("text", ""),
            )
            for index, chunk in enumerate(chunks, start=1)
        ]

        return AnswerResponse(
            question=question,
            answer=answer,
            sources=sources,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("DocuQuery answer generation failed")

        raise HTTPException(
            status_code=503,
            detail=(
                "Answer generation failed. Check the API "
                "terminal for the underlying error."
            ),
        ) from exc
