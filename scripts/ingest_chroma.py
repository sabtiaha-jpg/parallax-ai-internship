from pathlib import Path

import chromadb
import numpy as np
import pandas as pd


CHUNKS_PATH = Path("data/processed/embedded_chunks.parquet")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
CHROMA_PATH = "data/chroma_db"

COLLECTION_NAME = "wikipedia_chunks"
BATCH_SIZE = 5000


def ingest_chroma():
    print("=" * 55)
    print("WEEK 2 - CHROMADB INGESTION")
    print("=" * 55)

    print("\nLoading chunks and embeddings...")

    df = pd.read_parquet(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Chunks loaded: {len(df)}")
    print(f"Embeddings loaded: {len(embeddings)}")

    if len(df) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings does not match."
        )

    print("\nCreating persistent ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Remove old collection if the script is rerun
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("Old collection removed.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    print("Collection created successfully.")

    print("\nAdding data to ChromaDB...")

    total = len(df)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch_df = df.iloc[start:end]
        batch_embeddings = embeddings[start:end]

        ids = batch_df["chunk_id"].astype(str).tolist()

        documents = batch_df["text"].astype(str).tolist()

        metadatas = []

        for _, row in batch_df.iterrows():
            metadatas.append(
                {
                    "document_id": str(row["document_id"]),
                    "chunk_index": int(row["chunk_index"]),
                }
            )

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings.tolist(),
            metadatas=metadatas,
        )

        print(
            f"Inserted {end}/{total} chunks"
        )

    print("\nChromaDB ingestion complete!")

    print(
        f"Collection size: "
        f"{collection.count()}"
    )

    print(
        f"Database saved to: "
        f"{CHROMA_PATH}"
    )


if __name__ == "__main__":
    ingest_chroma()