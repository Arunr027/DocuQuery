
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of known relevant documents retrieved in the top k."""
    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top k results that are relevant."""
    if k <= 0:
        return 0.0

    top_results = retrieved_ids[:k]
    if not top_results:
        return 0.0

    return sum(item in relevant_ids for item in top_results) / len(top_results)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank

    return 0.0


def test_retrieval_metrics():
    retrieved = ["chunk_a", "chunk_b", "chunk_c"]
    relevant = {"chunk_b", "chunk_c"}

    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_retrieval_metrics_when_no_relevant_result_is_found():
    retrieved = ["chunk_a", "chunk_b"]
    relevant = {"chunk_c"}

    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert precision_at_k(retrieved, relevant, 2) == 0.0
    assert reciprocal_rank(retrieved, relevant) == 0.0
