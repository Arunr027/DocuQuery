\# DocuQuery — Enterprise Hybrid RAG \& Semantic Search Engine



DocuQuery is a document question-answering and semantic search service built with Python and FastAPI. It combines vector-based semantic search, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder reranking to retrieve relevant document passages and generate answers grounded in the available context.



\## Features



\- \*\*Document ingestion:\*\* Upload and process PDF and text documents.

\- \*\*Semantic search:\*\* Retrieve passages using embedding vectors and Qdrant.

\- \*\*Keyword search:\*\* Use BM25 to find passages containing relevant terms.

\- \*\*Hybrid retrieval:\*\* Combine semantic and keyword rankings using Reciprocal Rank Fusion.

\- \*\*Cross-encoder reranking:\*\* Reorder candidate passages according to their relevance to the question.

\- \*\*Grounded answers:\*\* Use the Gemini API to generate answers based on retrieved context.

\- \*\*Source references:\*\* Return source metadata with generated answers when available.

\- \*\*Redis caching:\*\* Cache hybrid search results to reduce repeated retrieval work.

\- \*\*REST API:\*\* Expose functionality through FastAPI and interactive Swagger documentation.

\- \*\*Containerized deployment:\*\* Run the API, Qdrant, and Redis using Docker Compose.

\- \*\*Automated tests:\*\* Validate API behavior, chunking, and retrieval evaluation metrics.



\## Architecture



```mermaid

flowchart TD

&#x20;   A\[Client / Swagger UI] --> B\[FastAPI]

&#x20;   B --> C\[Document Ingestion]

&#x20;   C --> D\[Text Extraction and Chunking]

&#x20;   D --> E\[Embeddings]

&#x20;   E --> F\[(Qdrant)]

&#x20;   B --> G\[Hybrid Retrieval]

&#x20;   G --> H\[Semantic Search]

&#x20;   G --> I\[BM25 Keyword Search]

&#x20;   H --> J\[Reciprocal Rank Fusion]

&#x20;   I --> J

&#x20;   J --> K\[Cross-Encoder Reranking]

&#x20;   K --> L\[Relevant Context]

&#x20;   L --> M\[Gemini Answer Generation]

&#x20;   M --> N\[Answer and Sources]

&#x20;   G <--> O\[(Redis Cache)]

```



\## Technology Stack



| Component | Technology |

|---|---|

| Language | Python |

| API framework | FastAPI |

| Vector database | Qdrant |

| Keyword retrieval | BM25 |

| Ranking fusion | Reciprocal Rank Fusion |

| Reranking | Sentence Transformers CrossEncoder |

| Language model | Google Gemini API |

| Cache | Redis |

| Testing | pytest |

| Deployment | Docker and Docker Compose |



\## Retrieval Pipeline



1\. Convert the user's question into an embedding.

2\. Retrieve candidate passages from Qdrant using semantic search.

3\. Retrieve keyword-matching passages using BM25.

4\. Merge rankings with Reciprocal Rank Fusion.

5\. Rerank the candidate passages with a cross-encoder.

6\. Diversify results across files and cache the final retrieval results in Redis.

7\. Use retrieved context to generate a grounded answer with Gemini.



The retrieval evaluation script compares semantic, keyword, and hybrid retrieval using Recall@5, Precision@5, and reciprocal rank. The current evaluation dataset is small and manually labelled; its results should be treated as preliminary rather than a general performance guarantee.



\## Requirements



\- Python 3.11 recommended

\- Docker Desktop with Docker Compose

\- A Google Gemini API key for answer generation



\## Run Locally



Clone or open the project directory, then create and activate a virtual environment.



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt

```



Create a `.env` file in the project root and configure the environment variables required by the application. Use the variable names defined in `app/core/config.py` and your existing environment configuration. Never commit API keys or secrets.



Start the API:



```powershell

python -m uvicorn app.main:app --reload

```



Open the interactive API documentation:



\- Swagger UI: http://localhost:8000/docs

\- Health check: http://localhost:8000/health



\## Run with Docker Compose



Make sure Docker Desktop is running. Configure the required environment variables in `.env`, then start the services:



```powershell

docker compose up -d

```



Check service status:



```powershell

docker compose ps

```



Check API health:



```powershell

Invoke-RestMethod http://localhost:8000/health

```



The standard local endpoints are:



\- API: http://localhost:8000

\- Swagger UI: http://localhost:8000/docs

\- Qdrant dashboard: http://localhost:6333/dashboard



Use the project's existing Docker Compose configuration and environment variables when deploying.



\## Testing



Run the automated test suite from the project root:



```powershell

python -m pytest -q

```



Latest verified result: \*\*18 passed\*\*.



The test suite includes API tests, document chunking tests, and retrieval evaluation metric tests. These tests do not, by themselves, establish production-scale retrieval accuracy or load performance.



\## Retrieval Evaluation



The evaluation script compares three retrieval approaches:



\- Semantic retrieval

\- BM25 keyword retrieval

\- Hybrid retrieval with fusion and reranking



Run the evaluation script with:



```powershell

python -m scripts.evaluate\_retrieval

```



Confirm the module path against the script's actual location if your project uses a different layout. Evaluation scores depend on the dataset, relevance labels, and document collection.



\## Security and Production Considerations



\- Keep `.env` and API credentials out of version control.

\- Validate uploaded files and enforce upload-size limits.

\- Apply authentication and authorization before exposing the service publicly.

\- Configure network access and persistent storage appropriately.

\- Expand the relevance-labelled evaluation dataset before making performance claims.

\- Add monitoring, load tests, and operational alerting before production deployment.



\## Project Status



The local automated test suite has passed 18 tests. The Docker Compose API, Qdrant, and Redis services were also verified running, and the API health endpoint returned a healthy status.



This is a portfolio project demonstrating document ingestion, retrieval-augmented generation, hybrid information retrieval, reranking, caching, API development, and containerized deployment.
