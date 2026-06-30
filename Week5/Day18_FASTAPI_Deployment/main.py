"""
main.py: FastAPI Model Serving Application
Week 4, Day 1: Model Serialization & FastAPI Architecture

This application loads a pre-trained machine learning model from disk
and exposes it through a REST API for inference requests.
"""

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# ── Load Model at Startup ─────────────────────────────────
# joblib.load() deserializes the pickled model object back into memory.
# This happens once when the server starts, not on every request,
# so subsequent predictions are fast.
MODEL_PATH = "final_loan_default_model.pkl"
model = joblib.load(MODEL_PATH)

print(f"Model loaded successfully from {MODEL_PATH}")

app = FastAPI(
    title="Loan Default Prediction API",
    description="Serves predictions from a trained Gradient Boosting classifier",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    """Defines the expected shape of incoming JSON payloads."""
    data: dict


@app.get("/health-check")
def health_check():
    """Simple endpoint to verify the API is running and reachable."""
    return {"status": "API is live"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Placeholder endpoint. Currently accepts a JSON payload and
    prints it to the terminal for verification. Full prediction
    logic using the loaded model will be implemented tomorrow.
    """
    print("Received prediction request payload:")
    print(request.data)
    return {"message": "Payload received successfully", "payload": request.data}