
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# 5. Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["test_size"],
    random_state=params["split_random_state"],
    stratify=y
)


# 6. Create model
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    random_state=params["random_state"]
)

# 7. Train ONLY on training data
model.fit(X_train, y_train)

# 8. Save model
joblib.dump(model, MODEL_PATH)

print("Training completed!")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of trees: {params['n_estimators']}")
print(f"Model saved to: {MODEL_PATH}")
