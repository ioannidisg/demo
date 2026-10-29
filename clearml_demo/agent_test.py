
from clearml import Task
import platform

task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Agent - First Remote Execution",
    reuse_last_task_id=False
)

# Send the task to the ClearML Agent
task.execute_remotely(
    queue_name="default",
    clone=False,
    exit_process=True
)

# This code runs when the Agent picks up the task
print("Hello from ClearML Agent!")
print("Running on:", platform.node())

result = 10 * 5
print("Result:", result)

task.get_logger().report_single_value(
    name="test_result",
    value=result
)

task.close()
