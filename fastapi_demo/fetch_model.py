
import hashlib
from pathlib import Path
from shutil import copy2

from clearml import InputModel

MODEL_ID = "f0b9ec07b616488ab1102bb3e63c0219"
EXPECTED_SHA256 = "e8a1da23f38287c9acb88a306568cd2f963c4c7a012fac5b56269b5b3f75a0d7"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "dvc_demo" / "model.joblib"

# Retrieve the registered model from ClearML.
model = InputModel(model_id=MODEL_ID)
source = Path(model.get_local_copy(raise_on_error=True))

# Check file integrity using SHA-256.
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if sha256(source) != EXPECTED_SHA256:
    raise RuntimeError("Downloaded model does not match expected SHA-256!")

TARGET.parent.mkdir(parents=True, exist_ok=True)

if TARGET.exists():
    if sha256(source) != sha256(TARGET):
        raise RuntimeError("Existing model differs from ClearML model!")
    print("Model already exists and matches ClearML.")
else:
    copy2(source, TARGET)
    print(f"Model copied to: {TARGET}")

print(f"Model SHA-256: {sha256(source)}")
