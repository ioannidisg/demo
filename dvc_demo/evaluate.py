
from pathlib import Path
import json

import joblib
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# 1. Define paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
PARAMS_PATH = PROJECT_DIR / "params.yaml"

# 2. Load parameters
with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# 3. Load dataset
df = pd.read_csv(DATA_PATH)

X = df[["temperature", "vibration"]]
y = df["failure"]

# 4. Reproduce the SAME train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Load trained model
model = joblib.load(MODEL_PATH)

# 6. Predict ONLY on test data
y_pred = model.predict(X_test)

# 7. Calculate test metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),
}

# 8. Save evaluation metrics
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("Test evaluation completed!")
print(f"Test samples: {len(X_test)}")
print(json.dumps(metrics, indent=4))
