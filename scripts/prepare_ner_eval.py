import pandas as pd
import json

source = "data/nlp/ner_evaluation_sample.csv"
output = "data/nlp/ner_evaluation_reviewed.csv"

df = pd.read_csv(source)
df["manual_entities"] = df["manual_entities"].astype("object")

annotations = {
    "doc_133_chunk_72": [
        {"text": "1853", "label": "DATE"},
        {"text": "Cuypers", "label": "PERSON"},
        {"text": "Amsterdam Centraal", "label": "FAC"},
        {"text": "1924", "label": "DATE"},
        {"text": "the Catholic Church", "label": "ORG"},
        {"text": "the International Eucharistic Congress", "label": "EVENT"},
        {"text": "Amsterdam", "label": "GPE"}
    ],
    "doc_26_chunk_64": [
        {"text": "The Gumby Show", "label": "WORK_OF_ART"},
        {"text": "US", "label": "GPE"},
        {"text": "1957–1967", "label": "DATE"},
        {"text": "Mio Mao", "label": "WORK_OF_ART"},
        {"text": "Italy", "label": "GPE"},
        {"text": "1974–2005", "label": "DATE"},
        {"text": "Morph", "label": "WORK_OF_ART"},
        {"text": "UK", "label": "GPE"},
        {"text": "1977–2000", "label": "DATE"},
        {"text": "Wallace and Gromit", "label": "WORK_OF_ART"},
        {"text": "1989", "label": "DATE"},
        {"text": "Jan Švankmajer", "label": "PERSON"},
        {"text": "Dimensions of Dialogue", "label": "WORK_OF_ART"},
        {"text": "Czechoslovakia", "label": "GPE"},
        {"text": "1982", "label": "DATE"},
        {"text": "The Trap Door", "label": "WORK_OF_ART"},
        {"text": "1984", "label": "DATE"},
        {"text": "Wallace & Gromit: The Curse of the Were-Rabbit", "label": "WORK_OF_ART"},
        {"text": "Chicken Run", "label": "WORK_OF_ART"},
        {"text": "The Adventures of Mark Twain", "label": "WORK_OF_ART"}
    ],
    "doc_104_chunk_131": [
        {"text": "Karnaugh", "label": "PERSON"},
        {"text": "Jevons", "label": "PERSON"},
        {"text": "1880", "label": "DATE"}
    ],
    "doc_13_chunk_22": [],
    "doc_177_chunk_1": []
}

for chunk_id, entities in annotations.items():
    df.loc[df["chunk_id"] == chunk_id, "manual_entities"] = json.dumps(
        entities, ensure_ascii=False
    )

df.to_csv(output, index=False, encoding="utf-8-sig")
print("Saved:", output)
print("Rows annotated:", sum(df["chunk_id"].isin(annotations)))
print(df.loc[df["chunk_id"].isin(annotations), 
             ["chunk_id", "manual_entities"]].to_string(index=False))
