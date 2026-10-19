from app.retrieval.cache import clear_search_cache
import hashlib

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.embeddings.embedder import embed_texts
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import extract_document
from app.retrieval.vector_store import (
    store_chunks,
    delete_previous_versions,
)
from app.schemas.documents import (
    ChunkPreview,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
):
    filename = (file.filename or "").strip()
    extension = (
        "." + filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    try:
        if not filename or extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported.",
            )

        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 10 MB upload limit.",
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        pages = extract_document(filename, content)

        if not pages:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No extractable text was found. "
                    "The PDF may be scanned or the document may be empty."
                ),
            )

        chunks = chunk_document(pages)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No usable text chunks were created.",
            )

        # Create an ID before embedding and storing.
        document_id = hashlib.sha256(content).hexdigest()

        try:
            vectors = embed_texts([chunk.text for chunk in chunks])

            if len(vectors) != len(chunks):
                raise ValueError(
                    "Each chunk must have exactly one embedding vector."
                )

            stored_count = store_chunks(
                document_id=document_id,
                filename=filename,
                chunks=chunks,
                vectors=vectors,
            )
            deleted_count = delete_previous_versions(
                filename=filename,
                keep_document_id=document_id,
            )
            clear_search_cache()
            print("Redis search cache cleared after document upload.")

        except Exception as exc:
            # Do not expose internal errors or secrets to API clients.
            raise HTTPException(
                status_code=503,
                detail="Document indexing failed. Check the API logs.",
            ) from exc

        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            pages_with_text=len(pages),
            chunks_created=len(chunks),
            stored_chunks=stored_count,
            total_characters=sum(len(page.text) for page in pages),
            chunks=[
                ChunkPreview(
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    character_count=len(chunk.text),
                    preview=chunk.text[:250],
                )
                for chunk in chunks
            ],
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()
