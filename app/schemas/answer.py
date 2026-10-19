
from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class SourceCitation(BaseModel):
    source_id: str
    filename: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    score: float | None = None
    text: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitation]
