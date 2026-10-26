from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Paths relative to the script location
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "model.joblib"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# 2. Separate features and target
X = df[["temperature", "vibration"]]
y = df["failure"]

# 3. Create model
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

# 4. Train model
model.fit(X, y)

# 5. Save trained model
joblib.dump(model, MODEL_PATH)

print(f"Training completed using {len(df)} samples.")
print(f"Model saved to: {MODEL_PATH}")