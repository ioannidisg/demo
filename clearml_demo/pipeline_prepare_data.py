
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.model_selection import train_test_split
from clearml import Task
import subprocess
import sys

Task.add_requirements("dvc==3.67.1")
Task.add_requirements("pandas")
Task.add_requirements("scikit-learn")

# 1. Locate the DVC-managed dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "dvc_demo" / "train.csv"

# 2. Create ClearML Task
task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Pipeline - Prepare Data",
    reuse_last_task_id=False,
    output_uri=True
)

# 3. Load dataset

# Retrieve dataset through DVC if it is missing
if not DATA_PATH.exists():
    print("Dataset missing. Running DVC pull...", flush=True)

    subprocess.run(
        [
            sys.executable, "-m", "dvc",
            "remote", "modify", "--local",
            "storage", "url",
            "C:/Users/ioann/dvc-storage"
        ],
        cwd=PROJECT_DIR,
        check=True
    )

    subprocess.run(
        [sys.executable, "-m", "dvc", "pull"],
        cwd=PROJECT_DIR,
        check=True
    )

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset could not be retrieved: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# 4. Split into training and test datasets
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["failure"]
)

# 5. Upload the resulting datasets as artifacts
with TemporaryDirectory() as tmp:
    train_path = Path(tmp) / "train_split.csv"
    test_path = Path(tmp) / "test_split.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    task.upload_artifact(
        name="train_dataset",
        artifact_object=str(train_path),
        wait_on_upload=True
    )

    task.upload_artifact(
        name="test_dataset",
        artifact_object=str(test_path),
        wait_on_upload=True
    )

print("Data preparation completed!")
print("Training samples:", len(train_df))
print("Test samples:", len(test_df))

task.close()
