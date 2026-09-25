import time

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "wikipedia_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def semantic_search(query: str):
    print("=" * 55)
    print("WEEK 2 - SEMANTIC SEARCH")
    print("=" * 55)

    print(f"\nQuery: {query}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(f"Collection size: {collection.count()}")

    print("\nSearching...")

    start_time = time.perf_counter()

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=TOP_K,
    )

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    print(f"\nSearch latency: {latency_ms:.2f} ms")

    print("\nTop results:")

    for i, document in enumerate(
        results["documents"][0],
        start=1
    ):
        distance = results["distances"][0][i - 1]

        print("\n" + "-" * 55)
        print(f"Result #{i}")
        print(f"Distance: {distance:.4f}")
        print(document)


if __name__ == "__main__":
    query = input("Enter a search query: ")

    semantic_search(query)