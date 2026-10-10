import json
import pandas as pd
from pathlib import Path

input_file = Path("data/nlp/ner_evaluation_reviewed.csv")
output_file = Path("data/nlp/ner_evaluation_metrics.json")

df = pd.read_csv(input_file)

def parse_entities(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {
        (item["text"].strip(), item["label"])
        for item in json.loads(value)
    }

tp = fp = fn = 0
evaluated = 0

print("\nNER Evaluation — Preliminary Sample")
print("-" * 45)

for _, row in df.iterrows():
    if pd.isna(row["manual_entities"]):
        continue

    predicted = parse_entities(row["named_entities"])
    gold = parse_entities(row["manual_entities"])

    tp += len(predicted & gold)
    fp += len(predicted - gold)
    fn += len(gold - predicted)
    evaluated += 1

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

results = {
    "evaluated_chunks": evaluated,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "evaluation_type": "Preliminary exact-match entity evaluation"
}

print(f"Chunks evaluated: {evaluated}")
print(f"True positives:   {tp}")
print(f"False positives:  {fp}")
print(f"False negatives:  {fn}")
print(f"Precision:        {precision:.4f}")
print(f"Recall:           {recall:.4f}")
print(f"F1-score:         {f1:.4f}")

output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nResults saved to: {output_file}")
print("IMPORTANT: Review the manual labels before reporting these scores.")
