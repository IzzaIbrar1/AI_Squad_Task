"""
main.py — FastAPI Model Serving Application
Week 4, Day 3 — OOD Guardrails & Postman Testing

Training dataset statistical boundaries used for OOD detection:
- Age: 21 to 70 (clipped during generation)
- Annual_Income: 15,000 to 250,000
- Loan_Amount: 1,000 to 73,511 (99th percentile cap applied)
- Credit_Score: 300 to 850 (standard FICO range, clipped)
- Debt_to_Income_Ratio: 0.01 to 0.95
- Employment_Years: 0 to 40
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Load Model at Startup ─────────────────────────────────
MODEL_PATH = "final_loan_default_model.pkl"
model = joblib.load(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")

# ── Initialize FastAPI Application ────────────────────────
app = FastAPI(
    title="Loan Default Prediction API",
    description="Real-time loan default prediction with OOD guardrails",
    version="3.0.0"
)


# ── Pydantic Schema ───────────────────────────────────────
class LoanApplication(BaseModel):
    Age: int
    Annual_Income: float
    Loan_Amount: float
    Credit_Score: int
    Employment_Years: float
    Debt_to_Income_Ratio: float
    Loan_Term_Months: int
    Num_Credit_Lines: int
    Has_CoSigner: int
    Previous_Defaults: int
    Home_Ownership_Own: int
    Home_Ownership_Rent: int
    Loan_Purpose_Business: int
    Loan_Purpose_Debt_Consolidation: int
    Loan_Purpose_Education: int
    Loan_Purpose_Home_Improvement: int
    Loan_Purpose_Medical: int
    Marital_Status_Married: int
    Marital_Status_Single: int
    Education_High_School: int
    Education_Master: int
    Education_PhD: int


# ── OOD Validation Function ───────────────────────────────
def validate_ood(data: LoanApplication):
    """
    Checks incoming values against the statistical boundaries of
    the training dataset. If any value falls outside the range
    the model was trained on, the model would be extrapolating
    into a feature space it has never optimized for — producing
    confident but meaningless predictions (silent failure).
    We intercept this before the data reaches the model.
    """
    if not (21 <= data.Age <= 70):
        raise HTTPException(
            status_code=400,
            detail=f"OOD Error: Age {data.Age} is outside training range (21–70)."
        )
    if not (15000 <= data.Annual_Income <= 250000):
        raise HTTPException(
            status_code=400,
            detail=f"OOD Error: Annual_Income {data.Annual_Income} is outside training range ($15,000–$250,000)."
        )
    if not (1000 <= data.Loan_Amount <= 73511):
        raise HTTPException(
            status_code=400,
            detail=f"OOD Error: Loan_Amount {data.Loan_Amount} is outside training range ($1,000–$73,511)."
        )
    if not (300 <= data.Credit_Score <= 850):
        raise HTTPException(
            status_code=400,
            detail=f"OOD Error: Credit_Score {data.Credit_Score} is outside valid FICO range (300–850)."
        )
    if not (0.0 <= data.Debt_to_Income_Ratio <= 0.95):
        raise HTTPException(
            status_code=400,
            detail=f"OOD Error: Debt_to_Income_Ratio {data.Debt_to_Income_Ratio} is outside training range (0.01–0.95)."
        )
    if not (0 <= data.Employment_Years <= 40):
        raise HTTPException(
            status_code=400,
            detail=f"OOD Error: Employment_Years {data.Employment_Years} is outside training range (0–40)."
        )


# ── Health Check Endpoint ─────────────────────────────────
@app.get("/health-check")
def health_check():
    return {"status": "API is live"}


# ── Predict Endpoint ──────────────────────────────────────
@app.post("/predict")
def predict(application: LoanApplication):
    """
    1. Pydantic validates data types (422 on wrong type)
    2. OOD guardrails check statistical boundaries (400 on out-of-range)
    3. Only clean, in-distribution data reaches the model
    """
    # Step 1: OOD check — runs before model sees any data
    validate_ood(application)

    # Step 2: Convert to DataFrame
    input_data = pd.DataFrame([application.model_dump()])

    # Step 3: Run prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "loan_default_prediction": int(prediction),
        "default_probability": round(float(probability), 4),
        "interpretation": "High risk of default" if prediction == 1 else "Low risk of default"
    }