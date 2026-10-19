
from app.retrieval.cache import get_cached_results, cache_results
from app.embeddings.embedder import embed_query
from app.retrieval.vector_store import search_chunks
from app.retrieval.keyword_search import search_keyword_chunks
from app.retrieval.reranker import rerank_chunks


def normalize_text(text: str) -> str:
    """Normalize whitespace and letter case for deduplication."""
    return " ".join(text.casefold().split())


def diversify_by_filename(
    results: list[dict],
    limit: int,
    max_per_file: int = 3,
) -> list[dict]:
    """Limit how many results come from the same file."""
    diversified = []
    filename_counts = {}

    for result in results:
        filename = result.get("filename") or "unknown"

        if filename_counts.get(filename, 0) >= max_per_file:
            continue

        diversified.append(result)
        filename_counts[filename] = filename_counts.get(filename, 0) + 1

        if len(diversified) >= limit:
            break

    return diversified


def search_hybrid_chunks(
    question: str,
    limit: int = 5,
) -> list[dict]:
    """Combine hybrid search, reranking, and Redis caching."""

    cached_results = get_cached_results(question, limit)

    if cached_results is not None:
        print("REDIS CACHE HIT")
        return cached_results

    print("REDIS CACHE MISS")

    candidate_limit = max(limit * 3, 10)
    query_vector = embed_query(question)

    semantic_results = search_chunks(
        query_vector=query_vector,
        limit=candidate_limit,
    )

    keyword_results = search_keyword_chunks(
        question=question,
        limit=candidate_limit,
    )

    # Combine rankings using Reciprocal Rank Fusion (RRF).
    rrf_k = 60
    combined = {}

    for ranked_results in (semantic_results, keyword_results):
        for rank, result in enumerate(ranked_results, start=1):
            text = result.get("text", "").strip()

            if not text:
                continue

            identity = normalize_text(text)

            if identity not in combined:
                combined[identity] = {
                    **result,
                    "score": 0.0,
                }

            combined[identity]["score"] += 1.0 / (rrf_k + rank)

    ranked_combined = sorted(
        combined.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    candidates = ranked_combined[:candidate_limit]

    # Rerank candidates before applying filename diversity.
    reranked_results = rerank_chunks(
        question=question,
        chunks=candidates,
        limit=candidate_limit,
    )

    results = diversify_by_filename(
        reranked_results,
        limit=limit,
        max_per_file=3,
    )

    cache_results(
        question=question,
        limit=limit,
        results=results,
    )

    return results
