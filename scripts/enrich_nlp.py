
import json
import pickle
import time
from pathlib import Path

import chromadb
import pandas as pd
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "wikipedia_chunks"

OUTPUT_DIR = Path("data/nlp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TOPICS = 10
TOP_WORDS = 8
BATCH_SIZE = 500
NER_BATCH_SIZE = 64
EVALUATION_SAMPLE_SIZE = 100
RANDOM_STATE = 42


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Connected to {COLLECTION_NAME}")
    print(f"Collection size: {collection.count()}")

    return collection


def load_documents(collection):
    records = collection.get(
        include=["documents", "metadatas"]
    )

    if not records["ids"]:
        raise ValueError("The ChromaDB collection is empty.")

    df = pd.DataFrame(
        {
            "chunk_id": records["ids"],
            "text": records["documents"],
            "metadata": records["metadatas"],
        }
    )

    df["text"] = df["text"].fillna("").astype(str)
    df["metadata"] = df["metadata"].apply(
        lambda value: value or {}
    )

    print(f"Loaded {len(df)} chunks.")
    print(f"Empty chunks: {(df['text'].str.strip() == '').sum()}")

    return df


def train_topic_model(df):
    print("\nTraining LDA topic model...")

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=10000,
        min_df=2,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    # Keep empty chunks valid without letting them break the model.
    texts = df["text"].tolist()
    valid_indices = [
        i for i, text in enumerate(texts)
        if len(text.strip()) > 0
    ]

    if not valid_indices:
        raise ValueError("No non-empty documents available.")

    matrix = vectorizer.fit_transform(
        [texts[i] for i in valid_indices]
    )

    if matrix.shape[1] == 0:
        raise ValueError("No usable vocabulary found for LDA.")

    n_topics = min(N_TOPICS, matrix.shape[0])

    model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method="batch",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    valid_distributions = model.fit_transform(matrix)
    feature_names = vectorizer.get_feature_names_out()

    # Assign -1 to empty documents so they are not mislabeled.
    topic_ids = [-1] * len(df)
    topic_names = ["Unassigned"] * len(df)

    topic_keywords = {}

    for topic_id, weights in enumerate(model.components_):
        top_indices = weights.argsort()[-TOP_WORDS:][::-1]
        words = feature_names[top_indices].tolist()

        topic_keywords[str(topic_id)] = words

    for row_position, original_index in enumerate(valid_indices):
        topic_id = int(
            valid_distributions[row_position].argmax()
        )

        topic_ids[original_index] = topic_id
        topic_names[original_index] = (
            " / ".join(topic_keywords[str(topic_id)][:3])
        )

    df["topic_id"] = topic_ids
    df["topic_name"] = topic_names

    with open(OUTPUT_DIR / "lda_model.pkl", "wb") as file:
        pickle.dump(
            {"model": model, "vectorizer": vectorizer},
            file,
        )

    with open(
        OUTPUT_DIR / "topic_keywords.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            topic_keywords,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Topics trained: {n_topics}")

    for topic_id, words in topic_keywords.items():
        print(f"Topic {topic_id}: {', '.join(words)}")

    return df


def extract_entities(df):
    print("\nLoading spaCy NER model...")
    nlp = spacy.load("en_core_web_sm")

    # Process only non-empty documents; preserve row order.
    valid_indices = [
        i for i, text in enumerate(df["text"])
        if text.strip()
    ]

    entities_by_index = [[] for _ in range(len(df))]

    texts = (df.iloc[i]["text"] for i in valid_indices)

    for index, doc in zip(
        valid_indices,
        nlp.pipe(texts, batch_size=NER_BATCH_SIZE),
    ):
        seen = set()
        entities = []

        for ent in doc.ents:
            entity_text = ent.text.strip()
            key = (entity_text.casefold(), ent.label_)

            if not entity_text or key in seen:
                continue

            seen.add(key)
            entities.append(
                {"text": entity_text, "label": ent.label_}
            )

        entities_by_index[index] = entities

    df["named_entities"] = [
        json.dumps(entities, ensure_ascii=False)
        for entities in entities_by_index
    ]

    # A simple readable field is useful for inspection.
    df["entity_texts"] = [
        ", ".join(dict.fromkeys(
            entity["text"] for entity in entities
        ))
        for entities in entities_by_index
    ]

    print("Named entity extraction complete.")

    return df


def create_evaluation_sample(df):
    print("\nCreating manual evaluation sample...")

    sample_size = min(EVALUATION_SAMPLE_SIZE, len(df))

    sample = df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).copy()

    sample_path = OUTPUT_DIR / "ner_evaluation_sample.csv"

    if sample_path.exists():
        print(
            f"Preserving existing evaluation file: {sample_path}"
        )
        return

    sample["manual_entities"] = ""

    sample[
        ["chunk_id", "text", "named_entities", "manual_entities"]
    ].to_csv(sample_path, index=False, encoding="utf-8-sig")

    print(f"Saved {sample_size} examples to {sample_path}")
    print(
        "Fill manual_entities with your verified gold labels "
        "before evaluating NER."
    )


def update_chroma(collection, df):
    print("\nUpdating existing ChromaDB metadata...")

    total = len(df)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]

        # Preserve existing metadata fields.
        metadatas = []

        for _, row in batch.iterrows():
            metadata = dict(row["metadata"])

            metadata.update(
                {
                    "topic_id": int(row["topic_id"]),
                    "topic_name": str(row["topic_name"]),
                    "named_entities": str(row["named_entities"]),
                    "entity_texts": str(row["entity_texts"]),
                }
            )

            metadatas.append(metadata)

        collection.update(
            ids=batch["chunk_id"].tolist(),
            metadatas=metadatas,
        )

        print(f"Updated metadata: {end}/{total}")

    print(f"Final collection size: {collection.count()}")


def main():
    start_time = time.perf_counter()

    collection = load_collection()
    df = load_documents(collection)

    df = train_topic_model(df)
    df = extract_entities(df)

    # Save results before updating the database.
    output_columns = [
        "chunk_id",
        "text",
        "topic_id",
        "topic_name",
        "named_entities",
        "entity_texts",
    ]

    df[output_columns].to_parquet(
        OUTPUT_DIR / "nlp_enriched_chunks.parquet",
        index=False,
    )

    create_evaluation_sample(df)
    update_chroma(collection, df)

    elapsed = time.perf_counter() - start_time

    print("\nNLP enrichment complete!")
    print(f"Total processing time: {elapsed:.2f} seconds")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
