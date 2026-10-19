
from pydantic import BaseModel


class ChunkPreview(BaseModel):
    chunk_index: int
    page_number: int | None
    character_count: int
    preview: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    pages_with_text: int
    chunks_created: int
    stored_chunks: int
    total_characters: int
    chunks: list[ChunkPreview]
