
import hashlib
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.core.config import settings


VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        timeout=10,
    )


def ensure_collection() -> None:
    client = get_qdrant_client()
    collection = settings.QDRANT_COLLECTION

    if not client.collection_exists(collection_name=collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def make_point_id(document_id: str, chunk_index: int) -> str:
    """Create a stable UUID-shaped ID for each document chunk."""
    from uuid import UUID

    value = f"{document_id}:{chunk_index}"
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()

    return str(UUID(digest))


def store_chunks(
    document_id: str,
    filename: str,
    chunks,
    vectors: list[list[float]],
) -> int:
    """Store chunks using stable point IDs."""

    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one vector.")

    ensure_collection()
    client = get_qdrant_client()
    points = []

    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=make_point_id(document_id, chunk.chunk_index),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                },
            )
        )

    if points:
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
            wait=True,
        )

    return len(points)


def search_chunks(
    query_vector: list[float],
    limit: int = 5,
) -> list[dict]:
    """Retrieve semantically relevant, non-duplicate chunks."""

    ensure_collection()
    client = get_qdrant_client()

    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=max(limit * 10, 50),
        with_payload=True,
    )

    results = []
    seen_chunks = set()

    for point in response.points:
        payload = point.payload or {}

        filename = payload.get("filename")
        chunk_index = payload.get("chunk_index")
        page_number = payload.get("page_number")
        text = payload.get("text", "")

        identity = (
            filename,
            chunk_index,
            page_number,
            " ".join(text.casefold().split()),
        )

        if identity in seen_chunks:
            continue

        seen_chunks.add(identity)

        results.append({
            "score": point.score,
            "document_id": payload.get("document_id"),
            "filename": filename,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "text": text,
        })

        if len(results) >= limit:
            break

    return results

def delete_previous_versions(
    filename: str,
    keep_document_id: str,
) -> int:
    """Delete older indexed versions of a file after its new version is stored."""
    ensure_collection()
    client = get_qdrant_client()

    old_point_ids = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename),
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            if payload.get("document_id") != keep_document_id:
                old_point_ids.append(point.id)

        if next_offset is None:
            break

        offset = next_offset

    if old_point_ids:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.PointIdsList(
                points=old_point_ids
            ),
            wait=True,
        )

    return len(old_point_ids)
