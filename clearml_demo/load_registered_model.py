
from clearml import InputModel
import joblib
import pandas as pd

# 1. Identify the registered model
MODEL_ID = "f0b9ec07b616488ab1102bb3e63c0219"

# 2. Retrieve model from ClearML Model Registry
input_model = InputModel(model_id=MODEL_ID)

# 3. Download or access cached model
model_path = input_model.get_local_copy(
    raise_on_error=True
)

print("Registered model retrieved!")
print("Model ID:", input_model.id)
print("Model path:", model_path)

# 4. Load the trained Random Forest
model = joblib.load(model_path)

# 5. Make a prediction
new_data = pd.DataFrame({
    "temperature": [85.0],
    "vibration": [7.5]
})

prediction = model.predict(new_data)[0]
probability = model.predict_proba(new_data)[0, 1]

print("Predicted failure:", prediction)
print(f"Failure probability: {probability:.2%}")
