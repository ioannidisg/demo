
from clearml import Task
from pathlib import Path
import subprocess
import sys
import shutil

task = Task.init(
    project_name="KIOS ML Preparation",
    task_name="Agent - DVC Environment Check",
    reuse_last_task_id=False
)

task.execute_remotely(
    queue_name="default",
    clone=False,
    exit_process=True
)

# Environment information
project_dir = Path(__file__).resolve().parent.parent

print("=== AGENT ENVIRONMENT ===")
print("Python:", sys.executable)
print("Project directory:", project_dir)
print("DVC executable:", shutil.which("dvc"))

print("\n=== DVC VERSION ===")
result = subprocess.run(
    [sys.executable, "-m", "dvc", "version"],
    cwd=project_dir,
    capture_output=True,
    text=True
)
print(result.stdout or result.stderr)

print("\n=== DATASET CHECK ===")
dataset = project_dir / "dvc_demo" / "train.csv"
print("Dataset exists:", dataset.exists())

print("\n=== DVC REMOTE CONFIG ===")
config = project_dir / ".dvc" / "config"
if config.exists():
    print(config.read_text(encoding="utf-8"))
else:
    print("DVC configuration not found")

task.close()
