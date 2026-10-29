
from clearml import Task

# 1. Get our existing successful experiment
template_task = Task.get_task(
    project_name="KIOS ML Preparation",
    task_name="Agent - Random Forest Hyperparameter Test"
)

# 2. Check the hyperparameter exists
param_name = "Training Parameters/n_estimators"
template_params = template_task.get_parameters()

if param_name not in template_params:
    raise ValueError(
        f"Parameter {param_name} not found in template Task"
    )

print("Template Task ID:", template_task.id)

# 3. Define the experiments
tree_values = [5, 10, 20, 50, 100]

# 4. Clone, configure and enqueue
for trees in tree_values:

    cloned_task = Task.clone(
        source_task=template_task,
        name=f"Agent - Auto Experiment - {trees} trees"
    )

    # Change only the number of trees
    cloned_task.set_parameter(
        name=param_name,
        value=trees
    )

    # Verify the configured value
    print(
        f"Configured: {trees} trees | "
        f"Task ID: {cloned_task.id}"
    )

    # Send task to the worker queue
    Task.enqueue(
        task=cloned_task,
        queue_name="default"
    )

    print(f"Enqueued: {trees} trees")

print("All 5 experiments submitted!")
