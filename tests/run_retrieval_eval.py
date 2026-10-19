
import json
from pathlib import Path

from app.retrieval.hybrid_search import search_hybrid_chunks


DATASET_PATH = Path(__file__).with_name("evaluation_dataset.json")
K = 3


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    recalls = []
    reciprocal_ranks = []

    for item in dataset:
        results = search_hybrid_chunks(item["question"], limit=K)

        retrieved_ids = [
            f'{result.get("filename")}#{result.get("chunk_index")}'
            for result in results
        ]
        relevant_ids = set(item["relevant_chunk_ids"])

        recall = recall_at_k(retrieved_ids, relevant_ids, K)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        recalls.append(recall)
        reciprocal_ranks.append(rr)

        print(f'\nQuestion: {item["question"]}')
        print(f"Relevant: {sorted(relevant_ids)}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Recall@{K}: {recall:.2f}")
        print(f"Reciprocal rank: {rr:.2f}")

    print("\n=== Aggregate metrics ===")
    print(f"Queries evaluated: {len(dataset)}")
    print(f"Mean Recall@{K}: {sum(recalls) / len(recalls):.2f}")
    print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.2f}")


if __name__ == "__main__":
    main()
