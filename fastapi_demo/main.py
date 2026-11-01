
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# 1. Locate the trained model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "dvc_demo" / "model.joblib"


# 2. Load model when FastAPI starts
@asynccontextmanager
async def lifespan(app: FastAPI):

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    app.state.model = joblib.load(MODEL_PATH)
    print("Random Forest model loaded successfully!")

    yield

    # Release model when server shuts down
    app.state.model = None


# 3. Create FastAPI application
app = FastAPI(
    title="KIOS ML Model Serving API",
    version="1.1.0",
    lifespan=lifespan
)


# 4. Input schema
class SensorInput(BaseModel):
    temperature: float
    vibration: float = Field(ge=0)

@app.get("/version")
def get_version():
    return {
        "api_version": app.version
    }


# 5. Health endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None
    }


# 6. Measurements endpoint
@app.post("/measurements")
def receive_measurements(data: SensorInput):
    return {
        "temperature": data.temperature,
        "vibration": data.vibration,
        "message": "Measurements received successfully"
    }


# Predict machine failure
@app.post("/predict")
def predict_failure(data: SensorInput):

    # Convert incoming measurements to DataFrame
    X = pd.DataFrame([{
        "temperature": data.temperature,
        "vibration": data.vibration
    }])

    # Use the already loaded Random Forest
    model = app.state.model

    prediction = int(model.predict(X)[0])

    # Find the probability of class 1 (failure)
    failure_index = list(model.classes_).index(1)
    probability = float(
        model.predict_proba(X)[0, failure_index]
    )

    return {
        "predicted_failure": prediction,
        "failure_probability": probability
    }
