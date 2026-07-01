"""
main.py: FastAPI Model Serving Application
Week 4, Day 2: Pydantic Validation & Live Model Inference
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "final_loan_default_model.pkl"
model = joblib.load(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")

app = FastAPI(
    title="Loan Default Prediction API",
    description="Real-time loan default prediction via trained Gradient Boosting model",
    version="2.0.0"
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

@app.get("/health-check")
def health_check():
    return {"status": "API is live"}


@app.post("/predict")
def predict(application: LoanApplication):
    """
    Accepts a strictly validated LoanApplication JSON payload.
    Converts it to a DataFrame matching the model's training features,
    runs prediction, and returns the result as JSON.
    """
    # Convert Pydantic model to DataFrame
    input_data = pd.DataFrame([application.model_dump()])

    # Run prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "loan_default_prediction": int(prediction),
        "default_probability": round(float(probability), 4),
        "interpretation": "High risk of default" if prediction == 1 else "Low risk of default"
    }