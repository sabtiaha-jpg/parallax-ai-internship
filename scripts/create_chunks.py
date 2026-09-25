from pathlib import Path
import pandas as pd

from src.chunking import chunk_text


INPUT_PATH = Path("data/processed/clean_corpus.parquet")
OUTPUT_PATH = Path("data/processed/chunks.parquet")

CHUNK_SIZE = 500
OVERLAP = 100


def find_text_column(df: pd.DataFrame) -> str:
    """
    Find the column containing document text.
    """

    possible_columns = [
        "text",
        "content",
        "document",
        "body",
        "clean_text",
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find a text column. Available columns: {list(df.columns)}"
    )


def create_chunks():
    print("=" * 50)
    print("WEEK 2 - DOCUMENT CHUNKING")
    print("=" * 50)

    print("\nLoading cleaned dataset...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Documents loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    text_column = find_text_column(df)

    print(f"Using text column: {text_column}")

    chunk_records = []

    print("\nCreating chunks...")

    for document_id, row in df.iterrows():

        text = row[text_column]

        if pd.isna(text):
            continue

        text = str(text).strip()

        if not text:
            continue

        chunks = chunk_text(
            text=text,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
        )

        for chunk_index, chunk in enumerate(chunks):

            chunk_records.append(
                {
                    "document_id": document_id,
                    "chunk_id": f"doc_{document_id}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    chunks_df = pd.DataFrame(chunk_records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    chunks_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nChunking complete!")

    print(f"Original documents: {len(df)}")
    print(f"Total chunks: {len(chunks_df)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Overlap: {OVERLAP}")

    if len(df) > 0:
        average_chunks = len(chunks_df) / len(df)

        print(
            f"Average chunks per document: "
            f"{average_chunks:.2f}"
        )

    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\nSample chunks:")

    print(
        chunks_df[
            ["chunk_id", "text"]
        ].head(3)
    )


if __name__ == "__main__":
    create_chunks()