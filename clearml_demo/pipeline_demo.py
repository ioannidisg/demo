
from clearml import PipelineController, Task

# Find our successful training Task
base_task = Task.get_task(
    project_name="KIOS ML Preparation",
    task_name="Agent - Random Forest Hyperparameter Test"
)

# Create Pipeline Controller
pipeline = PipelineController(
    name="Random Forest Training Pipeline",
    project="KIOS ML Preparation",
    version="1.0.0"
)

pipeline.set_default_execution_queue("default")

# Step 1: Train with 20 trees
pipeline.add_step(
    name="train_20_trees",
    base_task_id=base_task.id,
    parameter_override={
        "Training Parameters/n_estimators": 20
    }
)

# Step 2: Train with 50 trees
pipeline.add_step(
    name="train_50_trees",
    base_task_id=base_task.id,
    parents=["train_20_trees"],
    parameter_override={
        "Training Parameters/n_estimators": 50
    }
)

# Run controller locally; steps execute on Agent
pipeline.start_locally(run_pipeline_steps_locally=False)
