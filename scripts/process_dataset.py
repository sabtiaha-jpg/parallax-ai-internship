import pandas as pd
from pathlib import Path
import sys

sys.path.append("src")

from preprocessing import clean_text, is_english


RAW_FILE = Path("data/raw/wikipedia_raw.parquet")
OUTPUT_FILE = Path("data/processed/clean_corpus.parquet")


def main():
    print("Loading raw dataset...")

    df = pd.read_parquet(RAW_FILE)

    print("Original documents:", len(df))

    print("Cleaning text...")

    df["clean_text"] = df["text"].apply(clean_text)

    print("Filtering non-English and empty documents...")

    df = df[
        df["clean_text"].apply(is_english)
        & (df["clean_text"].str.len() > 0)
    ]

    # Keep only useful columns
    df = df[
        ["id", "title", "url", "clean_text"]
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n=== Processing Complete ===")
    print("Clean documents:", len(df))
    print("Saved to:", OUTPUT_FILE)

    print("\nExample cleaned document:")
    print("Title:", df.iloc[0]["title"])
    print(df.iloc[0]["clean_text"][:500])


if __name__ == "__main__":
    main()