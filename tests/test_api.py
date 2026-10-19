import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import query as query_api
from app.api import answers as answers_api


client = TestClient(app)


def sample_search_result():
    return {
        "score": 0.82,
        "rerank_score": 3.65,
        "document_id": "doc-123",
        "filename": "architecture.pdf",
        "chunk_index": 0,
        "page_number": 1,
        "text": "DocuQuery combines dense vector search with BM25 keyword retrieval.",
    }


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to DocuQuery"
    assert response.json()["health"] == "/health"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "DocuQuery API",
    }


def test_hybrid_search_success(monkeypatch):
    monkeypatch.setattr(
        query_api,
        "search_hybrid_chunks",
        lambda question, limit: [sample_search_result()],
    )

    response = client.post(
        "/query/hybrid",
        json={
            "question": "Explain hybrid retrieval",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "Explain hybrid retrieval"
    assert len(data["results"]) == 1
    assert data["results"][0]["filename"] == "architecture.pdf"
    assert data["results"][0]["rerank_score"] == 3.65


def test_semantic_search_success(monkeypatch):
    monkeypatch.setattr(
        query_api,
        "embed_query",
        lambda question: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        query_api,
        "search_chunks",
        lambda query_vector, limit: [sample_search_result()],
    )

    response = client.post(
        "/query/search",
        json={"question": "Explain vector search"},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_keyword_search_success(monkeypatch):
    monkeypatch.setattr(
        query_api,
        "search_keyword_chunks",
        lambda question, limit: [sample_search_result()],
    )

    response = client.post(
        "/query/keyword",
        json={"question": "Explain keyword search"},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


@pytest.mark.parametrize(
    "endpoint",
    [
        "/query/search",
        "/query/keyword",
        "/query/hybrid",
        "/query/answer",
    ],
)
def test_rejects_short_question(endpoint):
    response = client.post(
        endpoint,
        json={"question": "ab"},
    )

    assert response.status_code == 422



def test_answer_success_with_citation(monkeypatch):
    monkeypatch.setattr(
        answers_api,
        "search_hybrid_chunks",
        lambda question, limit: [sample_search_result()],
    )

    async def fake_generate_answer(question, chunks):
        return "DocuQuery combines vector search and keyword search."

    monkeypatch.setattr(
        answers_api,
        "generate_grounded_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/query/answer",
        json={"question": "How does hybrid retrieval work?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "vector search" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_id"] == "S1"
    assert data["sources"][0]["filename"] == "architecture.pdf"


def test_answer_when_no_documents_match(monkeypatch):
    monkeypatch.setattr(
        answers_api,
        "search_hybrid_chunks",
        lambda question, limit: [],
    )

    response = client.post(
        "/query/answer",
        json={"question": "What is in these documents?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert "could not find relevant information" in data["answer"].lower()


def test_answer_with_insufficient_evidence_returns_no_sources(monkeypatch):
    monkeypatch.setattr(
        answers_api,
        "search_hybrid_chunks",
        lambda question, limit: [sample_search_result()],
    )

    async def fake_generate_answer(question, chunks):
        return (
            "The provided documents do not contain information "
            "to answer this question."
        )

    monkeypatch.setattr(
        answers_api,
        "generate_grounded_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/query/answer",
        json={"question": "What is the capital of Japan?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert "do not contain information" in data["answer"].lower()


def test_answer_with_supporting_evidence_keeps_sources(monkeypatch):
    monkeypatch.setattr(
        answers_api,
        "search_hybrid_chunks",
        lambda question, limit: [sample_search_result()],
    )

    async def fake_generate_answer(question, chunks):
        return "DocuQuery combines dense vector search with BM25 [S1]."

    monkeypatch.setattr(
        answers_api,
        "generate_grounded_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/query/answer",
        json={"question": "Explain hybrid retrieval"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_id"] == "S1"
