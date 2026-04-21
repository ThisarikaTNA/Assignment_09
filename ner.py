import os
import json
from collections import Counter
import pandas as pd
from transformers import pipeline
import mlflow

# =========================
# MLflow Setup
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Transformers_Downstream_Tasks")

os.makedirs("artifacts/task2_ner", exist_ok=True)

# =========================
# DATA (FULL)
# =========================
ner_paragraphs = [
    "President Biden hosted Chancellor Merz in Washington to deliberate on trade agreements and energy security.",
    "Microsoft debuted the newest Surface laptop at a showcase in Seattle. Satya Nadella emphasized the integrated AI hardware during his presentation.",
    "The Federal Reserve adjusted borrowing costs yesterday. Jerome Powell cautioned about economic volatility in the coming months.",
    "Mark Zuckerberg confirmed that Meta will establish a research hub in Zurich, Switzerland. The initiative received backing from regional officials.",
    "Sony revealed advanced PlayStation peripherals at the Tokyo Game Show in Japan. The reveal generated buzz among industry critics.",
    "The World Economic Forum in Davos gathered executives from across the globe. Børge Brende started the session with a talk on global trade.",
    "Blue Origin sent a crewed capsule into orbit from Van Horn, Texas. Jeff Bezos monitored the flight sequence from mission control.",
    "Walmart grew its distribution network in Toronto, Canada. The corporation intends to recruit 1,500 additional staff by autumn.",
    "OpenAI reached a milestone in large language model reasoning. Sam Altman shared the results at a San Francisco symposium.",
    "Disney+ signed a deal with Lucasfilm to produce exclusive animated content. Bob Iger described it as a strategic move for the brand.",
    "Engineers at NASA and Caltech developed a prototype for Mars exploration. Dr. Sarah Chen directed the Caltech department.",
    "NVIDIA showcased its latest Blackwell chips at a developer conference in San Jose. Jensen Huang explained the architecture to a crowded audience."
]

ner_df = pd.DataFrame({"paragraph": ner_paragraphs})

# =========================
# MODEL
# =========================
model_name = "dslim/bert-base-NER"

ner1 = pipeline("ner", model=model_name, aggregation_strategy="simple", device=-1)
ner2 = pipeline("ner", model=model_name, aggregation_strategy="first", device=-1)

# =========================
# HELPERS
# =========================
def map_entity_label(entity_group):
    label = entity_group.upper()
    if label in ["PER", "PERSON"]:
        return "Person"
    elif label in ["ORG", "ORGANIZATION"]:
        return "Organization"
    elif label in ["LOC", "LOCATION"]:
        return "Location"
    else:
        return None

def run_ner(paragraphs, pipe):
    records = []
    for i, p in enumerate(paragraphs, start=1):
        preds = pipe(p)
        for pred in preds:
            mapped = map_entity_label(pred.get("entity_group", ""))
            if mapped:
                records.append({
                    "paragraph_id": i,
                    "paragraph": p,
                    "entity_text": pred.get("word", ""),
                    "entity_type": mapped,
                    "score": float(pred.get("score", 0.0))
                })

    df = pd.DataFrame(records)

    counts = Counter(df["entity_type"]) if len(df) > 0 else {}
    dist = {
        "Person": counts.get("Person", 0),
        "Organization": counts.get("Organization", 0),
        "Location": counts.get("Location", 0)
    }

    total = sum(dist.values())
    return df, dist, total

# =========================
# RUN 1
# =========================
df1, dist1, total1 = run_ner(ner_df["paragraph"].tolist(), ner1)

csv1 = "artifacts/task2_ner/task2_run1_entities.csv"
json1 = "artifacts/task2_ner/task2_run1_distribution.json"

df1.to_csv(csv1, index=False)
with open(json1, "w") as f:
    json.dump(dist1, f, indent=4)

with mlflow.start_run(run_name="Task2_Run1"):
    mlflow.log_param("task", "ner")
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("aggregation", "simple")
    mlflow.log_metric("total_entities", total1)
    mlflow.log_metric("person_count", dist1["Person"])
    mlflow.log_metric("organization_count", dist1["Organization"])
    mlflow.log_metric("location_count", dist1["Location"])
    mlflow.log_artifact(csv1)
    mlflow.log_artifact(json1)

# =========================
# RUN 2
# =========================
df2, dist2, total2 = run_ner(ner_df["paragraph"].tolist(), ner2)

csv2 = "artifacts/task2_ner/task2_run2_entities.csv"
json2 = "artifacts/task2_ner/task2_run2_distribution.json"

df2.to_csv(csv2, index=False)
with open(json2, "w") as f:
    json.dump(dist2, f, indent=4)

with mlflow.start_run(run_name="Task2_Run2"):
    mlflow.log_param("task", "ner")
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("aggregation", "first")
    mlflow.log_metric("total_entities", total2)
    mlflow.log_metric("person_count", dist2["Person"])
    mlflow.log_metric("organization_count", dist2["Organization"])
    mlflow.log_metric("location_count", dist2["Location"])
    mlflow.log_artifact(csv2)
    mlflow.log_artifact(json2)

print("NER completed")
