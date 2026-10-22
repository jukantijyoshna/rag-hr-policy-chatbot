
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "rag_db"

# Must be the same embedding model used during ingestion.
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_collection(
    name="hr_policies",
    embedding_function=None,
)


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """Retrieve the top-k chunks relevant to a question."""

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    # Do not request more results than the collection contains.
    result_count = min(top_k, collection.count())

    if result_count == 0:
        return []

    # Convert the question into a vector.
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()[0]

    # Search ChromaDB using that vector.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            {
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return retrieved


if __name__ == "__main__":
    question = input("Ask a question about the HR policies: ")

    matches = retrieve(question)

    if not matches:
        print("No documents available for retrieval.")
    else:
        print("\nRetrieved chunks:\n")

        for rank, match in enumerate(matches, start=1):
            print(f"--- Result {rank} ---")
            print(f"Text: {match['text']}")
            print(f"Source: {match['source']}")
            print(f"Distance: {match['distance']:.4f}")
            print()
