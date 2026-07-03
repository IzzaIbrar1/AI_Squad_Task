"""
main.py — FastAPI Model Serving Application

Adds a PredictionResponse Pydantic model to enforce a strict,
documented output schema. The endpoint no longer returns a raw
dict it returns a validated PredictionResponse object.
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "final_loan_default_model.pkl"
model = joblib.load(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")

app = FastAPI(
    title="Loan Default Prediction API",
    description="Real-time loan default prediction with OOD guardrails and structured response models",
    version="4.0.0"
)


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

class PredictionResponse(BaseModel):
    prediction: int
    confidence_score: float
    interpretation: str


def validate_ood(data: LoanApplication):
    if not (21 <= data.Age <= 70):
        raise HTTPException(status_code=400, detail=f"OOD Error: Age {data.Age} is outside training range (21-70).")
    if not (15000 <= data.Annual_Income <= 250000):
        raise HTTPException(status_code=400, detail=f"OOD Error: Annual_Income {data.Annual_Income} is outside training range ($15,000-$250,000).")
    if not (1000 <= data.Loan_Amount <= 73511):
        raise HTTPException(status_code=400, detail=f"OOD Error: Loan_Amount {data.Loan_Amount} is outside training range ($1,000-$73,511).")
    if not (300 <= data.Credit_Score <= 850):
        raise HTTPException(status_code=400, detail=f"OOD Error: Credit_Score {data.Credit_Score} is outside valid FICO range (300-850).")
    if not (0.0 <= data.Debt_to_Income_Ratio <= 0.95):
        raise HTTPException(status_code=400, detail=f"OOD Error: Debt_to_Income_Ratio {data.Debt_to_Income_Ratio} is outside training range (0.01-0.95).")
    if not (0 <= data.Employment_Years <= 40):
        raise HTTPException(status_code=400, detail=f"OOD Error: Employment_Years {data.Employment_Years} is outside training range (0-40).")


@app.get("/health-check")
def health_check():
    return {"status": "API is live"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    """
    Three-layer protection:
    1. Pydantic validates data types (422 on wrong type)
    2. OOD guardrails check statistical boundaries (400 on out-of-range)
    3. Only clean, in-distribution data reaches the model
    """
    validate_ood(application)

    input_data = pd.DataFrame([application.model_dump()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return PredictionResponse(
        prediction=int(prediction),
        confidence_score=round(float(probability), 4),
        interpretation="High risk of default" if prediction == 1 else "Low risk of default"
    )