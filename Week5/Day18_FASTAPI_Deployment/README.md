# Loan Default Prediction API
## Model Serialization & FastAPI Architecture

### Objective
Move the trained Loan Default model from a Jupyter Notebook into a production-style REST API using FastAPI, served locally with Uvicorn.

### What This Does
- Loads `final_loan_default_model.pkl` (Gradient Boosting classifier from the Weekend Project) into memory at server startup using `joblib`
- Exposes a health-check endpoint to verify the API is reachable
- Exposes a placeholder predict endpoint that accepts JSON and echoes it back (full prediction logic to be added next)

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health-check` | Returns `{"status": "API is live"}` |
| POST | `/predict` | Accepts validated LoanApplication JSON, returns prediction |

### Sample Response
```json
{
  "loan_default_prediction": 0,
  "default_probability": 0.1267,
  "interpretation": "Low risk of default"
}
```

### Validation
Invalid input (wrong types) automatically returns 422 Unprocessable Entity — the model never receives corrupted data.

### How to Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to access the Swagger UI and test both endpoints interactively.

### What is Model Serialization (Pickling)?
Pickling converts a Python object — in this case, a trained scikit-learn model with all its learned weights, tree structures, and parameters — into a byte stream that can be saved to disk and reloaded later without retraining. `joblib.dump()` performs this serialization; `joblib.load()` reverses it, reconstructing the exact same object in memory.

### Security Note
Loading an untrusted `.pkl` file from an external or unknown source is a serious security risk. Unlike JSON, the pickle format can execute arbitrary code during deserialization — a malicious actor could craft a `.pkl` file that runs harmful commands the moment it's loaded with `joblib.load()` or `pickle.load()`. Models should only be loaded from sources you trust and control, such as your own training pipeline.

### Files
| File | Description |
|------|--------------|
| `main.py` | FastAPI application |
| `final_loan_default_model.pkl` | Trained Gradient Boosting model |
| `requirements.txt` | Python dependencies |

---

## Day 20 Update: OOD Guardrails & Postman Testing

### Validation Layers (in order)
1. **Pydantic** — wrong data type → 422 automatically
2. **OOD Guardrails** — out-of-range values → 400 custom error
3. **Model** — only clean, in-distribution data reaches here

### Statistical Boundaries (based on training data)
| Feature | Min | Max |
|---------|-----|-----|
| Age | 21 | 70 |
| Annual_Income | $15,000 | $250,000 |
| Loan_Amount | $1,000 | $73,511 |
| Credit_Score | 300 | 850 |
| Debt_to_Income_Ratio | 0.01 | 0.95 |
| Employment_Years | 0 | 40 |

### Postman Test Result
Sending Age=-5 returned: `{"detail": "OOD Error: Age -5 is outside training range (21-70)."}`
Status: 400 Bad Request

---

## Day 21 Update: Response Models: Structured Prediction Output

### What Changed
- Added `PredictionResponse` Pydantic model defining the exact output schema
- `/predict` endpoint now uses `response_model=PredictionResponse` no raw dicts returned
- Swagger `/docs` now displays the response schema automatically

### Response Schema
```json
{
  "prediction": 0,
  "confidence_score": 0.1267,
  "interpretation": "Low risk of default"
}
```

### Full Validation Stack (all three layers active)
1. Pydantic request schema — wrong type → 422
2. OOD guardrails — out-of-range values → 400
3. Response model — output always matches defined schema