import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi_demo.main import app


def test_health():

    # Simulate an available, successfully loaded model
    with patch("fastapi_demo.main.MODEL_PATH") as mock_path:
        mock_path.is_file.return_value = True

        with patch("fastapi_demo.main.joblib.load",
                   return_value=object()):

            with TestClient(app) as client:
                response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True
    }


def test_predict_valid():

    # Create a fake Random Forest model
    fake_model = MagicMock()

    fake_model.classes_ = [0, 1]
    fake_model.predict.return_value = [1]
    fake_model.predict_proba.return_value = np.array([[0.45, 0.55]])

    # Mock model loading
    with patch("fastapi_demo.main.MODEL_PATH") as mock_path:
        mock_path.is_file.return_value = True

        with patch(
            "fastapi_demo.main.joblib.load",
            return_value=fake_model
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/predict",
                    json={
                        "temperature": 85.0,
                        "vibration": 7.5
                    }
                )

    # Verify the API response
    assert response.status_code == 200

    assert response.json() == {
        "predicted_failure": 1,
        "failure_probability": 0.55
    }


def test_predict_invalid_vibration():

    # Mock model loading during application startup
    with patch("fastapi_demo.main.MODEL_PATH") as mock_path:
        mock_path.is_file.return_value = True

        with patch(
            "fastapi_demo.main.joblib.load",
            return_value=object()
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/predict",
                    json={
                        "temperature": 85.0,
                        "vibration": -5.0
                    }
                )

    # Verify validation error
    assert response.status_code == 422

    # Verify which field caused the error
    errors = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "vibration"]
        for error in errors
    )


def test_predict_missing_temperature():

    # Mock model loading
    with patch("fastapi_demo.main.MODEL_PATH") as mock_path:
        mock_path.is_file.return_value = True

        with patch(
            "fastapi_demo.main.joblib.load",
            return_value=object()
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/predict",
                    json={"vibration": 7.5}
                )

    # Check validation error
    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "temperature"]
        for error in errors
    )
