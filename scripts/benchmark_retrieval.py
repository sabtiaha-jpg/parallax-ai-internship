from pathlib import Path
import time

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "wikipedia_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

OUTPUT_PATH = Path("results/retrieval_benchmark.csv")


QUERIES = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "What causes climate change?",
    "Who was Albert Einstein?",
    "What is the history of the internet?",
]


def run_benchmark():
    print("=" * 60)
    print("WEEK 2 - RETRIEVAL BENCHMARK")
    print("=" * 60)

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

    results_data = []

    print("\nRunning benchmark queries...\n")

    for query in QUERIES:

        print("-" * 60)
        print(f"Query: {query}")

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

        top_distance = results["distances"][0][0]

        top_result = results["documents"][0][0]

        print(f"Latency: {latency_ms:.2f} ms")
        print(f"Top distance: {top_distance:.4f}")
        print(f"Top result: {top_result[:150]}...")

        results_data.append(
            {
                "query": query,
                "latency_ms": latency_ms,
                "top_distance": top_distance,
                "top_result": top_result,
            }
        )

    benchmark_df = pd.DataFrame(results_data)

    average_latency = benchmark_df["latency_ms"].mean()
    min_latency = benchmark_df["latency_ms"].min()
    max_latency = benchmark_df["latency_ms"].max()

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"Queries tested: {len(benchmark_df)}")
    print(f"Average latency: {average_latency:.2f} ms")
    print(f"Minimum latency: {min_latency:.2f} ms")
    print(f"Maximum latency: {max_latency:.2f} ms")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    benchmark_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_benchmark()