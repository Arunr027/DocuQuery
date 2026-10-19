
from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the reranking model once and reuse it."""
    return CrossEncoder(MODEL_NAME)


def rerank_chunks(
    question: str,
    chunks: list[dict],
    limit: int = 5,
) -> list[dict]:
    """Reorder retrieved chunks by question relevance."""

    if not chunks:
        return []

    model = get_reranker()

    pairs = [
        (question, chunk.get("text", ""))
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    reranked = []

    for chunk, score in zip(chunks, scores):
        result = chunk.copy()
        result["rerank_score"] = float(score)
        reranked.append(result)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked[:limit]
