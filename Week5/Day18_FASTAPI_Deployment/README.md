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
| POST | `/predict` | Accepts a JSON payload, currently echoes it back |

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