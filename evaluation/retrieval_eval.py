"""Prepare human relevance labels and evaluate pooled Precision@K / Recall@K.

Run from repository root:
  python -m evaluation.retrieval_eval prepare
  # Edit evaluation/relevance_labels.json and set relevant_ids per query.
  python -m evaluation.retrieval_eval evaluate

Recall@K uses judged relevant documents *in the top-20 candidate pool*, not
all relevant documents in the full corpus. Do not call it corpus-wide recall.
"""
import argparse
import json
from pathlib import Path

from scripts import rag_pipeline as rag

BASE = Path(__file__).resolve().parent
QUESTIONS = BASE / "questions.json"
LABELS = BASE / "relevance_labels.json"
OUT = Path("results/retrieval_evaluation.json")
POOL_SIZE = 20
K = 5


def search_ids(question: str, size: int) -> list[dict]:
    embedding = rag.embedding_model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    result = rag.collection.query(
        query_embeddings=[embedding.tolist()], n_results=size,
        include=["documents", "distances"]
    )
    return [
        {"id": doc_id, "distance": round(float(dist), 4), "preview": (doc or "")[:220]}
        for doc_id, doc, dist in zip(
            result["ids"][0], result["documents"][0], result["distances"][0]
        )
    ]


def prepare() -> None:
    if LABELS.exists():
        raise SystemExit(f"{LABELS} already exists. Move/rename it before regenerating to avoid losing labels.")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    dataset = []
    for case in questions:
        candidates = search_ids(case["question"], POOL_SIZE)
        dataset.append({"question": case["question"], "relevant_ids": None, "candidates": candidates})
    LABELS.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {LABELS}. Review candidate previews (and full Chroma documents if needed); replace each null relevant_ids with a list of truly relevant candidate IDs, e.g. [\"id1\"]. An empty [] means none are relevant in this pool.")


def evaluate() -> None:
    if not LABELS.exists():
        raise SystemExit("Run 'prepare' first, then manually label relevant_ids.")
    cases = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        relevant = case.get("relevant_ids")
        if relevant is None:
            raise SystemExit(f"Unlabeled question: {case['question']}. Fill relevant_ids in {LABELS}.")
        candidate_ids = {x["id"] for x in case["candidates"]}
        if not isinstance(relevant, list) or not set(relevant).issubset(candidate_ids):
            raise SystemExit(f"Invalid relevance IDs for {case['question']}")
        top_ids = [x["id"] for x in search_ids(case["question"], K)]
        hits = len(set(top_ids) & set(relevant))
        # divide by requested K, not smaller actual result count
        precision = hits / K
        recall = hits / len(relevant) if relevant else None
        rows.append({"question": case["question"], "precision_at_5": precision,
                     "pooled_recall_at_5": recall, "relevant_in_pool": len(relevant), "hits": hits})
    recalls = [r["pooled_recall_at_5"] for r in rows if r["pooled_recall_at_5"] is not None]
    output = {"k": K, "candidate_pool": POOL_SIZE, "number_of_queries": len(rows),
              "mean_precision_at_5": sum(r["precision_at_5"] for r in rows) / len(rows),
              "mean_pooled_recall_at_5": sum(recalls) / len(recalls) if recalls else None,
              "note": "Recall denominator is manually judged relevant documents among top-20 candidates, NOT all corpus relevant documents. Null recalls excluded from mean.",
              "queries": rows}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "evaluate"])
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate}[args.action]()
