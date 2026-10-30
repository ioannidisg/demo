# 1. Python environment
FROM python:3.12-slim

# 2. Working directory inside container
WORKDIR /app

# 3. Install Python dependencies
COPY fastapi_demo/requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt

# 4. Copy FastAPI application
COPY fastapi_demo/main.py ./fastapi_demo/main.py

# 5. Copy trained Random Forest model
COPY dvc_demo/model.joblib ./dvc_demo/model.joblib

# 6. Create non-root user
RUN useradd --create-home appuser
USER appuser

# 7. Document application port
EXPOSE 8000

# 8. Start FastAPI server
CMD ["python", "-m", "uvicorn", "fastapi_demo.main:app", "--host", "0.0.0.0", "--port", "8000"]
