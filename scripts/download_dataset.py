from datasets import load_dataset
import pandas as pd
from pathlib import Path
from itertools import islice

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading Wikipedia dataset...")

# Stream Wikipedia so we do not download the entire dataset.
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)

print("Collecting 6,000 Wikipedia documents...")

records = []

for article in islice(dataset, 6000):
    records.append({
        "id": article.get("id"),
        "url": article.get("url"),
        "title": article.get("title"),
        "text": article.get("text")
    })

df = pd.DataFrame(records)

output_file = OUTPUT_DIR / "wikipedia_raw.parquet"

df.to_parquet(
    output_file,
    index=False
)

print("\n=== Dataset Download Complete ===")
print("Saved to:", output_file)
print("Number of documents:", len(df))
print("Columns:", df.columns.tolist())

print("\nExample article:")
print("Title:", df.iloc[0]["title"])
print(df.iloc[0]["text"][:500])