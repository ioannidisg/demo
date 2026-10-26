from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

# 1. Define paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
PARAMS_PATH = PROJECT_DIR / "params.yaml"

# 2. Load training parameters
with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# 3. Load dataset
df = pd.read_csv(DATA_PATH)

# 4. Separate features and target
X = df[["temperature", "vibration"]]
y = df["failure"]

# 5. Create model using YAML parameters
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    random_state=params["random_state"]
)

# 6. Train model
model.fit(X, y)

# 7. Save model
joblib.dump(model, MODEL_PATH)

print(f"Training completed using {len(df)} samples.")
print(f"Number of trees: {params['n_estimators']}")
print(f"Random state: {params['random_state']}")
print(f"Model saved to: {MODEL_PATH}")