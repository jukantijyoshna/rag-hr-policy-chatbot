
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "hr_policy.txt"
DB_DIR = BASE_DIR / "rag_db"

# 2. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def split_into_chunks(text: str) -> list[str]:
    """Split text into non-empty, paragraph-based chunks."""
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


def ingest_documents():
    # 3. Read the source document
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Document not found: {DATA_FILE}")

    text = DATA_FILE.read_text(encoding="utf-8")
    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError("The document contains no usable text.")

    print(f"Created {len(chunks)} chunks.")

    # 4. Generate embeddings
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
    ).tolist()

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Vector dimensions: {len(embeddings[0])}")

    # 5. Store chunks and embeddings in ChromaDB
    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name="hr_policies",
        embedding_function=None,
    )

    collection.upsert(
        ids=[f"hr_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": DATA_FILE.name,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ],
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    print(f"Database location: {DB_DIR}")


if __name__ == "__main__":
    ingest_documents()
