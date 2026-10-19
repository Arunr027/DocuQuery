
import re

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.retrieval.vector_store import (
    ensure_collection,
    get_qdrant_client,
)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase searchable terms."""
    return re.findall(r"\b[\w.-]+\b", text.lower())


def search_keyword_chunks(
    question: str,
    limit: int = 5,
) -> list[dict]:
    """Search stored document chunks using BM25."""

    ensure_collection()
    client = get_qdrant_client()

    all_points = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        all_points.extend(points)

        if next_offset is None:
            break

        offset = next_offset

    documents = []
    seen_texts = set()

    for point in all_points:
        payload = point.payload or {}
        text = payload.get("text", "").strip()

        if not text:
            continue

        identity = text.casefold()

        if identity in seen_texts:
            continue

        seen_texts.add(identity)

        documents.append({
            "document_id": payload.get("document_id"),
            "filename": payload.get("filename"),
            "chunk_index": payload.get("chunk_index"),
            "page_number": payload.get("page_number"),
            "text": text,
        })

    if not documents:
        return []

    tokenized_documents = [
        tokenize(document["text"])
        for document in documents
    ]

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        result = documents[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

        if len(results) >= limit:
            break

    return results
