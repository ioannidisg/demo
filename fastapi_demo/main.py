
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="KIOS ML Model Serving API")


# Schema for incoming sensor measurements
class SensorInput(BaseModel):
    temperature: float
    vibration: float = Field(ge=0)


# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}


# Receive and validate sensor measurements
@app.post("/measurements")
def receive_measurements(data: SensorInput):
    return {
        "temperature": data.temperature,
        "vibration": data.vibration,
        "message": "Measurements received successfully"
    }
