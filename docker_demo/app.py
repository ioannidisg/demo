import os
import numpy as np

# Read configuration from environment variables
model_version = os.getenv("MODEL_VERSION", "v1")
app_env = os.getenv("APP_ENV", "development")

data = np.array([10, 20, 30, 40, 50])

print("Docker ML Engineering Demo")
print("Model Version:", model_version)
print("Environment:", app_env)
print("Mean:", np.mean(data))
print("Standard Deviation:", np.std(data))
print("Testing Docker layer caching!")