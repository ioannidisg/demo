
from pathlib import Path
from clearml import Task, OutputModel

# 1. Locate the existing model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "dvc_demo" / "model.joblib"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

# 2. Create a ClearML Task
task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Register Random Forest Model - Cloud",
    reuse_last_task_id=False,
    output_uri=True
)

# 3. Create a registered Output Model
output_model = OutputModel(
    task=task,
    name="Random Forest - Failure Prediction",
    framework="scikit-learn"
)

# 4. Register the trained model file
output_model.update_weights(
    weights_filename=str(MODEL_PATH),
    auto_delete_file=False,
    async_enable=False
)

# 5. Finish
print("Model registered successfully!")
print("Model ID:", output_model.id)

task.close()
