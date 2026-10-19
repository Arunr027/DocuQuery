
import re
from dataclasses import dataclass

from app.ingestion.loader import ExtractedPage


@dataclass
class DocumentChunk:
    text: str
    chunk_index: int
    page_number: int | None


def clean_text(text: str) -> str:
    """Clean common PDF extraction whitespace artifacts."""

    # Replace non-breaking spaces and normalize line endings.
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words broken across lines by PDF extraction.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    # Collapse repeated whitespace and newlines.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split cleaned text near word boundaries with overlap."""

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Prefer a word boundary near the desired chunk end.
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(text):
            break

        next_start = max(0, end - overlap)

        # Guarantee forward progress.
        start = max(next_start, start + 1)

    return chunks


def chunk_document(
    pages: list[ExtractedPage],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Clean extracted pages and split them into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks: list[DocumentChunk] = []

    for page in pages:
        text = clean_text(page.text)

        if not text:
            continue

        page_chunks = split_into_chunks(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_text in page_chunks:
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    page_number=page.page_number,
                )
            )

    return chunks
