from clearml import Task

task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Connection Test",
    reuse_last_task_id=False
)

print("ClearML connection works!")

task.close()
