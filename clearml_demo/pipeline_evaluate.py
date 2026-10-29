
from clearml import Task
import pandas as pd
import joblib
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Create ClearML Evaluation Task
task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Pipeline - Evaluate Model",
    reuse_last_task_id=False,
    output_uri=True
)


# Parameters supplied by the Pipeline Controller
params = {
    "prepare_task_id": "",
    "train_task_id": ""
}

params = task.connect(
    params,
    name="Pipeline Parameters"
)


# 2. Retrieve the Data Preparation Task

# Retrieve Task IDs from pipeline parameters
prepare_task_id = params["prepare_task_id"]
train_task_id = params["train_task_id"]

if not prepare_task_id or not train_task_id:
    raise ValueError(
        "Missing prepare_task_id or train_task_id"
    )

# Retrieve the exact Tasks from this pipeline run
prepare_task = Task.get_task(
    task_id=prepare_task_id
)

train_task = Task.get_task(
    task_id=train_task_id
)

print("Prepare Task ID:", prepare_task_id)
print("Train Task ID:", train_task_id)

# 4. Download test dataset and trained model
test_path = prepare_task.artifacts[
    "test_dataset"
].get_local_copy()

model_path = train_task.artifacts[
    "trained_model"
].get_local_copy()

print("Test dataset retrieved:", test_path)
print("Trained model retrieved:", model_path)

# 5. Load model and test data
df = pd.read_csv(test_path)
model = joblib.load(model_path)

X_test = df[["temperature", "vibration"]]
y_test = df["failure"]

# 6. Evaluate
predictions = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    )
}

# 7. Report metrics to ClearML
logger = task.get_logger()

for name, value in metrics.items():
    logger.report_single_value(name=name, value=value)

# 8. Save metrics as an artifact
task.upload_artifact(
    name="evaluation_metrics",
    artifact_object=metrics,
    wait_on_upload=True
)

print("Evaluation completed!")
print("Test samples:", len(df))
print(json.dumps(metrics, indent=4))

task.close()
