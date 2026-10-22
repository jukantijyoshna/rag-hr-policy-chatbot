
from rag import retrieve

evaluation_data = [
    {
        "question": "How many annual leave days do employees receive?",
        "relevant_ids": ["hr_chunk_0"],
    },
    {
        "question": "How many days can employees work remotely?",
        "relevant_ids": ["hr_chunk_1"],
    },
    {
        "question": "How do employees apply for leave?",
        "relevant_ids": ["hr_chunk_2"],
    },
    {
        "question": "How many unused annual leave days can be carried forward?",
        "relevant_ids": ["hr_chunk_3"],
    },
]


def evaluate_retrieval(top_k=3):
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for test in evaluation_data:
        question = test["question"]
        relevant_ids = set(test["relevant_ids"])

        results = retrieve(question, top_k=top_k)

        # These labels assume each test has at least one relevant chunk.
        retrieved_ids = {
            f"hr_chunk_{item['chunk_index']}"
            for item in results
        }

        relevant_found = len(relevant_ids & retrieved_ids)

        precision = (
            relevant_found / len(retrieved_ids)
            if retrieved_ids else 0.0
        )

        recall = relevant_found / len(relevant_ids)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        total_precision += precision
        total_recall += recall
        total_f1 += f1

        print(f"\nQuestion: {question}")
        print(f"Relevant chunks found: {relevant_found}")
        print(f"Precision@{top_k}: {precision:.2f}")
        print(f"Recall@{top_k}: {recall:.2f}")
        print(f"F1@{top_k}: {f1:.2f}")

    count = len(evaluation_data)

    print("\n===== AVERAGE RESULTS =====")
    print(f"Precision@{top_k}: {total_precision / count:.2f}")
    print(f"Recall@{top_k}: {total_recall / count:.2f}")
    print(f"F1@{top_k}: {total_f1 / count:.2f}")


if __name__ == "__main__":
    evaluate_retrieval(top_k=1)
