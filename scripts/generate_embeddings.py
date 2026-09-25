from pathlib import Path
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/chunks.parquet")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
SELECTED_CHUNKS_PATH = Path("data/processed/embedded_chunks.parquet")

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
SAMPLE_SIZE = 20000


def generate_embeddings():
    print("=" * 55)
    print("WEEK 2 - EMBEDDING GENERATION")
    print("=" * 55)

    print("\nLoading chunks...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Total chunks available: {len(df)}")

    # Use a smaller subset so embedding is practical on CPU
    df = df.head(SAMPLE_SIZE).copy()

    print(f"Chunks selected for embedding: {len(df)}")

    # Remove empty text rows just in case
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].copy()

    print(f"Valid non-empty chunks: {len(df)}")

    texts = df["text"].tolist()

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")

    print("\nGenerating embeddings...")
    print(f"Batch size: {BATCH_SIZE}")

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    end_time = time.perf_counter()

    total_time = end_time - start_time

    print("\nEmbedding generation complete!")

    print(f"Number of embeddings: {len(embeddings)}")

    if len(embeddings) > 0:
        print(f"Embedding dimension: {embeddings.shape[1]}")

        average_time = total_time / len(embeddings)
        throughput = len(embeddings) / total_time

        print(f"Total embedding time: {total_time:.2f} seconds")
        print(
            f"Average embedding time per chunk: "
            f"{average_time * 1000:.4f} ms"
        )
        print(
            f"Embedding throughput: "
            f"{throughput:.2f} chunks/second"
        )

    # Save embeddings
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    # Save the exact chunks that match these embeddings
    df.to_parquet(
        SELECTED_CHUNKS_PATH,
        index=False
    )

    print(f"\nEmbeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Selected chunks saved to: {SELECTED_CHUNKS_PATH}")

    print("\nDone!")


if __name__ == "__main__":
    generate_embeddings()