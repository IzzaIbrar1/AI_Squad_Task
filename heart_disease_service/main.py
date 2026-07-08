"""
Heart Disease Risk Prediction Microservice
FastAPI wrapper around a Gradient Boosting classifier.

Run with: uvicorn main:app --reload
Docs at: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk from clinical patient data.",
    version="1.0.0",
)

# Load model once at startup
artifact = joblib.load("model.pkl")
model = artifact["model"]
FEATURE_COLUMNS = artifact["feature_columns"]

# ---------------------------------------------------------------------------
# OOD (Out-of-Distribution) clinical guardrails.
# Pydantic enforces hard physiological bounds (see PatientData below) and
# returns 422 if those are violated. TYPICAL_RANGES is a second, tighter
# check: values that pass Pydantic but fall outside the range actually seen
# in training data are rejected with a 400, since a prediction on such input
# would be an extrapolation the model wasn't trained to make reliably.
# ---------------------------------------------------------------------------
TYPICAL_RANGES = {
    "age": (29, 77),
    "trestbps": (94, 200),   # resting blood pressure, mm Hg
    "chol": (126, 564),      # serum cholesterol, mg/dl
    "thalach": (71, 202),    # max heart rate achieved
    "oldpeak": (0.0, 6.2),   # ST depression induced by exercise
}


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=60, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type (0-3)")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = low risk, 1 = high risk")
    risk_label: str
    probability: float = Field(..., description="Model's predicted probability of heart disease")
    confidence: str


def check_ood(data: PatientData) -> list[str]:
    """Returns a list of field names that fall outside TYPICAL_RANGES."""
    flagged = []
    for field, (low, high) in TYPICAL_RANGES.items():
        value = getattr(data, field)
        if value < low or value > high:
            flagged.append(f"{field}={value} (expected {low}-{high})")
    return flagged


@app.get("/", summary="Root", tags=["Health"])
def root():
    """Basic liveness check,  confirms the service is running."""
    return {"status": "ok", "service": "Heart Disease Risk Prediction API"}


@app.get("/health", summary="Health check", tags=["Health"])
def health():
    """Confirms the service is up and the trained model loaded successfully."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict heart disease risk",
    tags=["Prediction"],
)
def predict(patient: PatientData):
    """
    Takes 13 clinical measurements and returns a heart disease risk prediction.

    - Returns **422** if a field is missing, has the wrong type, or fails a
      hard physiological bound (e.g. negative age).
    - Returns **400** if a value is technically valid but far outside the
      range the model was trained on (out-of-distribution input).
    - Returns **200** with `prediction`, `probability`, and `confidence`
      otherwise.
    """
    ood_violations = check_ood(patient)
    if ood_violations:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Input out of distribution",
                "message": "One or more values fall outside the clinical range seen in training data. "
                            "A prediction on this input would be unreliable.",
                "violations": ood_violations,
            },
        )

    try:
        input_df = pd.DataFrame([patient.model_dump()])[FEATURE_COLUMNS]

        pred = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[0][1])

        if proba >= 0.75 or proba <= 0.25:
            confidence = "high"
        elif proba >= 0.6 or proba <= 0.4:
            confidence = "moderate"
        else:
            confidence = "low"

        return PredictionResponse(
            prediction=pred,
            risk_label="high risk" if pred == 1 else "low risk",
            probability=round(proba, 4),
            confidence=confidence,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
