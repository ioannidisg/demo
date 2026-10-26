from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

X = df[["temperature", "vibration"]]
y_true = df["failure"]

# 2. Load trained model
model = joblib.load(MODEL_PATH)

# 3. Generate predictions
y_pred = model.predict(X)

# 4. Calculate evaluation metrics
metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
}

# 5. Save metrics to JSON
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation completed!")
print(json.dumps(metrics, indent=4))