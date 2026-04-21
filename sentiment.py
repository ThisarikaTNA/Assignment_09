import os
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import pipeline
import mlflow

warnings.filterwarnings("ignore")

# =========================
# MLflow Setup
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Transformers_Downstream_Tasks")

# =========================
# Create folder
# =========================
os.makedirs("artifacts/task1_sentiment", exist_ok=True)

# =========================
# DATA (FULL)
# =========================
sentiment_data = [
    {"text": "The kitchen was messy and had a foul odor.", "true_label": "NEGATIVE"},
    {"text": "I thoroughly enjoyed this concert. It was spectacular throughout.", "true_label": "POSITIVE"},
    {"text": "The delivery was delayed and the contents were broken.", "true_label": "NEGATIVE"},
    {"text": "This camera captures beautifully and the lens is top-notch.", "true_label": "POSITIVE"},
    {"text": "The software glitches constantly whenever I run a task.", "true_label": "NEGATIVE"},
    {"text": "The technical assistance person was extremely efficient and polite.", "true_label": "POSITIVE"},
    {"text": "I feel cheated by this purchase. It stopped working instantly.", "true_label": "NEGATIVE"},
    {"text": "This bakery produces incredible pastries every single day.", "true_label": "POSITIVE"},
    {"text": "The interface is frustrating and difficult to navigate.", "true_label": "NEGATIVE"},
    {"text": "The speaker conveyed the message effectively and with passion.", "true_label": "POSITIVE"},
    {"text": "I am let down by how subpar the materials are.", "true_label": "NEGATIVE"},
    {"text": "The latest patch made the game more responsive and stable.", "true_label": "POSITIVE"},
    {"text": "The hotel was disgusting and the staff were incredibly rude.", "true_label": "NEGATIVE"},
    {"text": "Our trip was peaceful, exciting, and worth every penny.", "true_label": "POSITIVE"},
    {"text": "This was an absolute loss of time and resources.", "true_label": "NEGATIVE"},
    {"text": "The poem was moving and elegantly composed.", "true_label": "POSITIVE"},
    {"text": "I disliked the final chapter of the novel. It felt rushed.", "true_label": "NEGATIVE"},
    {"text": "I appreciated the lecture and gained valuable insights.", "true_label": "POSITIVE"},
    {"text": "The speakers sound distorted and look very flimsy.", "true_label": "NEGATIVE"},
    {"text": "I am truly impressed with the durability of this watch.", "true_label": "POSITIVE"}
]

df = pd.DataFrame(sentiment_data)

# =========================
# MODEL
# =========================
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name, device=-1)

# =========================
# RUN 1
# =========================
preds1 = classifier(df["text"].tolist(), truncation=True)

run1_df = df.copy()
run1_df["predicted_label"] = [p["label"] for p in preds1]
run1_df["confidence_score"] = [p["score"] for p in preds1]

acc1 = accuracy_score(run1_df["true_label"], run1_df["predicted_label"])

csv1 = "artifacts/task1_sentiment/task1_run1_predictions.csv"
run1_df.to_csv(csv1, index=False)

with mlflow.start_run(run_name="Task1_Run1"):
    mlflow.log_param("model", model_name)
    mlflow.log_metric("accuracy", acc1)
    mlflow.log_artifact(csv1)

print("Task1 Run1 Accuracy:", acc1)

# =========================
# RUN 2
# =========================
preds2 = classifier(df["text"].tolist(), truncation=True, max_length=32)

run2_df = df.copy()
run2_df["predicted_label"] = [p["label"] for p in preds2]
run2_df["confidence_score"] = [p["score"] for p in preds2]

acc2 = accuracy_score(run2_df["true_label"], run2_df["predicted_label"])

csv2 = "artifacts/task1_sentiment/task1_run2_predictions.csv"
run2_df.to_csv(csv2, index=False)

with mlflow.start_run(run_name="Task1_Run2"):
    mlflow.log_param("model", model_name)
    mlflow.log_param("max_length", 32)
    mlflow.log_metric("accuracy", acc2)
    mlflow.log_artifact(csv2)

print("Task1 Run2 Accuracy:", acc2)
