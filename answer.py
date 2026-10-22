
import ollama
from rag import retrieve

MAX_DISTANCE = 0.85


def answer_question(question: str) -> str:
    # 1. Retrieve candidate chunks
    matches = retrieve(question, top_k=3)

    # 2. Keep only chunks within our provisional threshold
    relevant_matches = [
        item for item in matches
        if item["distance"] <= MAX_DISTANCE
    ]

    # 3. Abstain if no sufficiently close chunk exists
    if not relevant_matches:
        return (
            "I don't have enough relevant information in the "
            "HR policy documents to answer that question."
        )

    # 4. Build context from accepted chunks only
    context = "\n\n".join(
        item["text"] for item in relevant_matches
    )

    # 5. Generate an answer using the local LLM
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an HR policy assistant. "
                    "Answer using only the supplied context. "
                    "If the context does not contain enough "
                    "information, say so. Never invent policy details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"REFERENCE CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}"
                ),
            },
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    while True:
        question = input("\nAsk an HR policy question (or type 'exit'): ")

        if question.strip().lower() == "exit":
            break

        if not question.strip():
            continue

        try:
            print("\nAnswer:", answer_question(question))
        except Exception as error:
            print(f"\nError: {error}")
