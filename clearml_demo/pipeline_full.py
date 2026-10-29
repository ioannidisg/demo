
from clearml import PipelineController

# Existing ClearML Task Templates
PREPARE_TEMPLATE_ID = "7d471b12320d492bbf618fffc3469596"
TRAIN_TEMPLATE_ID = "99e48ffb26624a4d8183306a17285017"
EVALUATE_TEMPLATE_ID = "ba1ab7c4b5c24397b1f0450981bcaaee"

# 1. Create the Pipeline Controller
pipeline = PipelineController(
    name="KIOS - Complete ML Pipeline",
    project="KIOS ML Preparation",
    version="1.0.0"
)

# 2. Set the Agent execution queue
pipeline.set_default_execution_queue("default")

# 3. Step 1: Prepare Data
pipeline.add_step(
    name="prepare_data",
    base_task_id=PREPARE_TEMPLATE_ID
)

# 4. Step 2: Train Model
pipeline.add_step(
    name="train_model",
    base_task_id=TRAIN_TEMPLATE_ID,
    parents=["prepare_data"],
    parameter_override={
        "Pipeline Parameters/prepare_task_id":
            "${prepare_data.id}",
        "Pipeline Parameters/n_estimators": 20
    }
)

# 5. Step 3: Evaluate Model
pipeline.add_step(
    name="evaluate_model",
    base_task_id=EVALUATE_TEMPLATE_ID,
    parents=["prepare_data", "train_model"],
    parameter_override={
        "Pipeline Parameters/prepare_task_id":
            "${prepare_data.id}",
        "Pipeline Parameters/train_task_id":
            "${train_model.id}"
    }
)

# 6. Run the controller locally
# Pipeline steps will execute through ClearML Agent
pipeline.start_locally(
    run_pipeline_steps_locally=False
)
