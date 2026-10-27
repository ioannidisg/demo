
from clearml import Task
import joblib
import pandas as pd

# 1. Task that contains the uploaded model
TASK_ID = "0571111350c44c24a61b72c1b44d64b0"

# 2. Retrieve the existing ClearML Task
previous_task = Task.get_task(task_id=TASK_ID)

# 3. Download model artifact
model_path = previous_task.artifacts[
    "trained_random_forest"
].get_local_copy()

print("Model retrieved from ClearML!")
print("Cached model path:", model_path)

# 4. Load trained Random Forest
model = joblib.load(model_path)

# 5. Create a new sensor measurement
new_data = pd.DataFrame({
    "temperature": [85.0],
    "vibration": [7.5]
})

# 6. Make a prediction
prediction = model.predict(new_data)[0]
probability = model.predict_proba(new_data)[0, 1]

print("Predicted failure:", prediction)
print(f"Failure probability: {probability:.2%}")
