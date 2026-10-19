
from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.answers import router as answers_router


app = FastAPI(
    title="DocuQuery API",
    description=(
        "Document ingestion, semantic search, "
        "and grounded RAG answers with citations."
    ),
    version="0.4.0",
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(answers_router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to DocuQuery",
        "docs": "/docs",
        "health": "/health",
    }
