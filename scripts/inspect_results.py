
from app.embeddings.embedder import embed_query
from app.retrieval.vector_store import search_chunks
from app.retrieval.keyword_search import search_keyword_chunks
from app.retrieval.hybrid_search import search_hybrid_chunks

question = "Which vector databases and orchestration tools are listed?"

methods = {
    "Semantic": lambda: search_chunks(
        query_vector=embed_query(question), limit=5
    ),
    "Keyword": lambda: search_keyword_chunks(question, limit=5),
    "Hybrid": lambda: search_hybrid_chunks(question, limit=5),
}

for method_name, search in methods.items():
    print(f"\n{'=' * 20} {method_name} {'=' * 20}")

    for rank, result in enumerate(search(), start=1):
        print(
            f"\nRank {rank} | "
            f"{result.get('filename')} | "
            f"Chunk {result.get('chunk_index')}"
        )
        print(result.get("text", "")[:250])
