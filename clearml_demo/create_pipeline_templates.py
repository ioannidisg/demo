
from pathlib import Path
import subprocess
from clearml import Task

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Get the exact Git commit
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=PROJECT_DIR,
    text=True
).strip()

# GitHub repository
REPO_URL = "https://github.com/ioannidisg/demo.git"

def create_template(name, script, packages, parameters=None):
    task = Task.create(
        project_name="KIOS ML Preparation",
        task_name=name,
        repo=REPO_URL,
        branch="feature/clearml-tracking",
        commit=commit,
        script=f"clearml_demo/{script}",
        working_directory=".",
        packages=["clearml==2.1.12", *packages],
        add_task_init_call=False
    )

    if parameters:
        task.set_parameters(parameters)

    print(f"{name}: {task.id}")
    return task


prepare = create_template(
    name="Template - Prepare Data",
    script="pipeline_prepare_data.py",
    packages=[
        "dvc==3.67.1",
        "pandas",
        "scikit-learn"
    ]
)

train = create_template(
    name="Template - Train Model",
    script="pipeline_train.py",
    packages=[
        "pandas",
        "scikit-learn",
        "joblib"
    ],
    parameters={
        "Pipeline Parameters/prepare_task_id": "",
        "Pipeline Parameters/n_estimators": "20"
    }
)

evaluate = create_template(
    name="Template - Evaluate Model",
    script="pipeline_evaluate.py",
    packages=[
        "pandas",
        "scikit-learn",
        "joblib"
    ],
    parameters={
        "Pipeline Parameters/prepare_task_id": "",
        "Pipeline Parameters/train_task_id": ""
    }
)

print("All pipeline templates created!")
