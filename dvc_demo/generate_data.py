
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "train.csv"

# Reproducible synthetic data
rng = np.random.default_rng(42)

n_samples = 1000

# Generate sensor measurements
temperature = rng.uniform(35, 110, n_samples)
vibration = rng.uniform(0.1, 10.0, n_samples)

# Failure probability increases with sensor values
logits = (
    -4.0
    + 0.045 * (temperature - 35)
    + 0.35 * vibration
)

probability = 1 / (1 + np.exp(-logits))

# Sample binary failures
failure = rng.binomial(1, probability)

# Create dataset
df = pd.DataFrame({
    "temperature": temperature.round(2),
    "vibration": vibration.round(3),
    "failure": failure
})

df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset created: {OUTPUT_PATH}")
print(f"Total samples: {len(df)}")
print("\nClass distribution:")
print(df["failure"].value_counts())
print("\nFirst five rows:")
print(df.head())
