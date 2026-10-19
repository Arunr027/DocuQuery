
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass
class ExtractedPage:
    text: str
    page_number: int | None


def extract_document(
    filename: str,
    content: bytes,
) -> list[ExtractedPage]:
    """Extract text while preserving PDF page numbers."""
    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        return _extract_pdf(content)

    if extension == "txt":
        return _extract_txt(content)

    raise ValueError("Only PDF and TXT files are supported.")


def _extract_pdf(content: bytes) -> list[ExtractedPage]:
    try:
        reader = PdfReader(BytesIO(content))

        if reader.is_encrypted:
            raise ValueError(
                "Encrypted PDFs are not supported."
            )

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                pages.append(
                    ExtractedPage(
                        text=text,
                        page_number=page_number,
                    )
                )

        return pages

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Could not read this PDF. Check that it is valid."
        ) from exc


def _extract_txt(content: bytes) -> list[ExtractedPage]:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(
            "TXT files must use UTF-8 encoding."
        ) from exc

    if not text:
        return []

    return [ExtractedPage(text=text, page_number=None)]
