
from clearml import Task
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Create the training Task
task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Pipeline - Train Model",
    reuse_last_task_id=False,
    output_uri=True
)


# Parameters supplied by the Pipeline Controller
params = {
    "prepare_task_id": "",
    "n_estimators": 20
}

params = task.connect(
    params,
    name="Pipeline Parameters"
)



prepare_task_id = params["prepare_task_id"]

if not prepare_task_id:
    raise ValueError("Missing prepare_task_id")

prepare_task = Task.get_task(
    task_id=prepare_task_id
)


# 3. Retrieve its training artifact
train_path = prepare_task.artifacts[
    "train_dataset"
].get_local_copy()

print("Training dataset retrieved:", train_path)

# 4. Load the training data
df = pd.read_csv(train_path)

X_train = df[["temperature", "vibration"]]
y_train = df["failure"]

# 5. Train Random Forest
model = RandomForestClassifier(
    n_estimators=int(params["n_estimators"]),
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed!")
print("Training samples:", len(df))
print("Number of trees:", len(model.estimators_))

# 6. Upload trained model as artifact
with TemporaryDirectory() as tmp:
    model_path = Path(tmp) / "model.joblib"
    joblib.dump(model, model_path)

    task.upload_artifact(
        name="trained_model",
        artifact_object=str(model_path),
        wait_on_upload=True
    )

print("Trained model uploaded to ClearML!")

task.close()
