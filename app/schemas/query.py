
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    score: float
    rerank_score: float | None = None
    document_id: str | None
    filename: str | None
    chunk_index: int | None
    page_number: int | None
    text: str


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResult]
