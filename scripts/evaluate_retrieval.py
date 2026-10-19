
from app.embeddings.embedder import embed_query
from app.retrieval.vector_store import search_chunks
from app.retrieval.keyword_search import search_keyword_chunks
from app.retrieval.hybrid_search import search_hybrid_chunks

DOC_A = "dfcf2882-64b9-46d9-9c14-dac87bbbfbea"
DOC_B = "f5f5b1b98fb9a97a4c73206a75b6a410c2767bee88d67d22208578129b3fc9dc"

EVALUATION_DATASET = [
    {
        "question": "Explain DocuQuery's hybrid retrieval architecture.",
        "relevant": {"Arun GenAI.pdf:3"},
    },
    {
        "question": "How does the agent runtime handle tool execution and session persistence?",
        "relevant": {"Arun GenAI.pdf:4"},
    },
    {
        "question": "What metrics does the LLM observability dashboard monitor?",
        "relevant": {"Arun GenAI.pdf:2"},
    },
    {
        "question": "Which vector databases and orchestration tools are listed?",
        "relevant": {"Arun GenAI.pdf:0", "Arun GenAI.pdf:1"},
    },
]

def chunk_id(result):
    return f"{result.get('filename')}:{result.get('chunk_index')}"

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def precision_at_k(retrieved, relevant, k):
    if k <= 0:
        return 0.0
    return sum(item in relevant for item in retrieved[:k]) / k

def reciprocal_rank(retrieved, relevant):
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0

def evaluate():
    methods = {"Semantic": [], "Keyword": [], "Hybrid": []}

    for case in EVALUATION_DATASET:
        question = case["question"]
        relevant = case["relevant"]

        print(f"\nQuestion: {question}")

        vector = embed_query(question)

        results_by_method = {
            "Semantic": search_chunks(query_vector=vector, limit=5),
            "Keyword": search_keyword_chunks(question=question, limit=5),
            "Hybrid": search_hybrid_chunks(question=question, limit=5),
        }

        for method, results in results_by_method.items():
            retrieved = [chunk_id(result) for result in results]

            recall = recall_at_k(retrieved, relevant, 5)
            precision = precision_at_k(retrieved, relevant, 5)
            mrr = reciprocal_rank(retrieved, relevant)

            methods[method].append((recall, precision, mrr))

            print(
                f"  {method:8} | "
                f"Recall@5={recall:.2f} | "
                f"Precision@5={precision:.2f} | "
                f"RR={mrr:.2f}"
            )

    print("\n===== AVERAGE RESULTS =====")

    for method, scores in methods.items():
        count = len(scores)
        avg_recall = sum(score[0] for score in scores) / count
        avg_precision = sum(score[1] for score in scores) / count
        avg_mrr = sum(score[2] for score in scores) / count

        print(
            f"{method:8} | "
            f"Recall@5={avg_recall:.3f} | "
            f"Precision@5={avg_precision:.3f} | "
            f"MRR={avg_mrr:.3f}"
        )

if __name__ == "__main__":
    evaluate()
