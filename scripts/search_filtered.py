
import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "wikipedia_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(MODEL_NAME)

    keywords_path = Path("data/nlp/topic_keywords.json")
    with open(keywords_path, encoding="utf-8") as file:
        topics = json.load(file)

    print("\nAvailable topics:")
    for topic_id, words in topics.items():
        print(f"{topic_id}: {', '.join(words)}")

    query = input("\nEnter your search query: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    topic_input = input(
        "Enter a topic ID to filter, or press Enter for all topics: "
    ).strip()

    where = None

    if topic_input:
        try:
            topic_id = int(topic_input)
        except ValueError:
            print("Topic ID must be an integer.")
            return

        if str(topic_id) not in topics:
            print("Unknown topic ID.")
            return

        where = {"topic_id": topic_id}

    start = time.perf_counter()

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    kwargs = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": TOP_K,
        "include": ["documents", "metadatas", "distances"],
    }

    if where is not None:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    elapsed_ms = (time.perf_counter() - start) * 1000

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nResults returned: {len(documents)}")
    print(f"Search latency: {elapsed_ms:.2f} ms")

    if not documents:
        print("No matching documents found.")
        return

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        print("\n" + "-" * 60)
        print(f"Result #{rank}")
        print(f"Distance: {distance:.4f}")
        print(f"Topic ID: {metadata.get('topic_id')}")
        print(f"Topic: {metadata.get('topic_name')}")
        print(f"Entities: {metadata.get('entity_texts', '')}")
        print(f"Text: {document}")


if __name__ == "__main__":
    main()
