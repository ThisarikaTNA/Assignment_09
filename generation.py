import os
import pandas as pd
from transformers import pipeline
import mlflow

# =========================
# MLflow Setup
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Transformers_Downstream_Tasks")

os.makedirs("artifacts/task3_text_generation", exist_ok=True)

# =========================
# DATA (FULL)
# =========================
prompts = [
    "The secret to sustainable energy lies in",
    "Beyond the furthest reaches of our galaxy,",
    "The most significant cultural shift of this century is",
    "Inside the abandoned laboratory, a flickering screen revealed",
    "The way we communicate will be fundamentally altered by",
]

df = pd.DataFrame({"prompt": prompts})

# =========================
# MODEL
# =========================
model_name = "gpt2"
generator = pipeline("text-generation", model=model_name, device=-1)

# =========================
# FUNCTION
# =========================
def generate(prompts, temp, max_len):
    records = []
    for i, p in enumerate(prompts, start=1):
        out = generator(
            p,
            max_length=max_len,
            temperature=temp,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=50256
        )[0]["generated_text"]

        records.append({
            "prompt_id": i,
            "prompt": p,
            "generated_text": out
        })

    return pd.DataFrame(records)

# =========================
# RUN 1
# =========================
temp1, len1 = 0.7, 50
df1 = generate(df["prompt"].tolist(), temp1, len1)

csv1 = "artifacts/task3_text_generation/task3_run1_generated_outputs.csv"
txt1 = "artifacts/task3_text_generation/task3_run1_observations.txt"

df1.to_csv(csv1, index=False)

with open(txt1, "w") as f:
    f.write(f"""
Run1 Observations:
Temperature={temp1}, MaxLength={len1}
Balanced output, moderate creativity.
""")

with mlflow.start_run(run_name="Task3_Run1"):
    mlflow.log_param("task", "text_generation")
    mlflow.log_param("temperature", temp1)
    mlflow.log_param("max_length", len1)
    mlflow.log_artifact(csv1)
    mlflow.log_artifact(txt1)

# =========================
# RUN 2
# =========================
temp2, len2 = 1.0, 70
df2 = generate(df["prompt"].tolist(), temp2, len2)

csv2 = "artifacts/task3_text_generation/task3_run2_generated_outputs.csv"
txt2 = "artifacts/task3_text_generation/task3_run2_observations.txt"

df2.to_csv(csv2, index=False)

with open(txt2, "w") as f:
    f.write(f"""
Run2 Observations:
Temperature={temp2}, MaxLength={len2}
More creative but less controlled outputs.
""")

with mlflow.start_run(run_name="Task3_Run2"):
    mlflow.log_param("task", "text_generation")
    mlflow.log_param("temperature", temp2)
    mlflow.log_param("max_length", len2)
    mlflow.log_artifact(csv2)
    mlflow.log_artifact(txt2)

print("Generation completed")
