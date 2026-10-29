
from clearml import Task
from pathlib import Path
import subprocess
import sys
import pandas as pd

# Required packages for the Agent
Task.add_requirements("dvc")
Task.add_requirements("pandas")

task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Agent - DVC Pull Test",
    reuse_last_task_id=False
)

task.execute_remotely(
    queue_name="default",
    clone=False,
    exit_process=True
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 1. Configure the local storage path for this checkout
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

# 2. Verify the remote configuration
subprocess.run(
    [sys.executable, "-m", "dvc", "remote", "list"],
    cwd=PROJECT_DIR,
    check=True
)

# 3. Retrieve DVC data and outputs
subprocess.run(
    [sys.executable, "-m", "dvc", "pull"],
    cwd=PROJECT_DIR,
    check=True
)

# 4. Verify the dataset
DATASET = PROJECT_DIR / "dvc_demo" / "train.csv"

print("Dataset exists:", DATASET.exists())

if DATASET.exists():
    df = pd.read_csv(DATASET)
    print("Dataset shape:", df.shape)
    print(df.head())

task.close()
