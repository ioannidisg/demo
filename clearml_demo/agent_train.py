
from clearml import Task
from pathlib import Path
import subprocess
import sys
import os
import json

# Dependencies for the Agent's Python environment
Task.add_requirements("dvc==3.67.1")
Task.add_requirements("pandas")
Task.add_requirements("scikit-learn")
Task.add_requirements("joblib")
Task.add_requirements("PyYAML")

task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Agent - Random Forest Training",
    reuse_last_task_id=False
)

# Submit experiment to the ClearML worker
task.execute_remotely(
    queue_name="default",
    clone=False,
    exit_process=True
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_DIR / "dvc_demo" / "metrics.json"

# Ensure DVC stages use the worker's Python
env = os.environ.copy()
env["PATH"] = (
    str(Path(sys.executable).parent)
    + os.pathsep
    + env.get("PATH", "")
)


def run_dvc(*args):
    subprocess.run(
        [sys.executable, "-m", "dvc", *args],
        cwd=PROJECT_DIR,
        env=env,
        check=True
    )


# 1. Configure local DVC remote
run_dvc(
    "remote", "modify", "--local",
    "storage", "url",
    "C:/Users/ioann/dvc-storage"
)

# 2. Retrieve dataset
print("Pulling DVC data...", flush=True)
run_dvc("pull")

# 3. Run training and evaluation
print("Running ML pipeline...", flush=True)
run_dvc("repro", "--force")

# 4. Read metrics
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# 5. Report metrics to ClearML
logger = task.get_logger()

for name, value in metrics.items():
    logger.report_single_value(
        name=name,
        value=value
    )

print("Evaluation metrics:", metrics)
print("Agent training completed!")

task.close()
