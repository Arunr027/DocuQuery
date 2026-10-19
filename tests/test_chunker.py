
import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import ExtractedPage


def test_short_text_creates_one_chunk():
    pages = [
        ExtractedPage(
            text="Database connection troubleshooting.",
            page_number=1,
        )
    ]

    chunks = chunk_document(pages)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert "Database connection" in chunks[0].text


def test_long_text_creates_overlapping_chunks():
    text = "A" * 2500
    pages = [ExtractedPage(text=text, page_number=2)]

    chunks = chunk_document(
        pages,
        chunk_size=1000,
        overlap=200,
    )

    assert len(chunks) == 3
    assert chunks[0].text[-200:] == chunks[1].text[:200]
    assert chunks[1].text[-200:] == chunks[2].text[:200]


def test_invalid_chunk_configuration():
    pages = [ExtractedPage(text="Some text", page_number=1)]

    with pytest.raises(ValueError):
        chunk_document(pages, chunk_size=100, overlap=100)
