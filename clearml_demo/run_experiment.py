
from pathlib import Path
import json
import subprocess

import yaml
from clearml import Task

# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

PARAMS_PATH = PROJECT_DIR / "params.yaml"
METRICS_PATH = PROJECT_DIR / "dvc_demo" / "metrics.json"

# 2. Load hyperparameters
with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# 3. Create ClearML experiment
task = Task.init(
    project_name="KIOS ML Preparation",
    task_name=f"Random Forest - {params['n_estimators']} trees -v2 -test split",
    reuse_last_task_id=False
)

# 4. Track hyperparameters
task.connect(params, name="Training Parameters")

# 5. Execute existing DVC pipeline
print("Starting DVC pipeline...")

subprocess.run(
    ["dvc", "repro", "--force"],
    cwd=PROJECT_DIR,
    check=True
)

# 6. Read real evaluation results
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# 7. Track evaluation metrics
logger = task.get_logger()

for metric_name, value in metrics.items():
    logger.report_single_value(
        name=metric_name,
        value=value
    )

    logger.report_scalar(
        title="Evaluation Metrics",
        series=metric_name,
        value=value,
        iteration=1
    )

print("Evaluation metrics:", metrics)

# 8. Complete experiment
task.close()

print("ClearML experiment completed!")
