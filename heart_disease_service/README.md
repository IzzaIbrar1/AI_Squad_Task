# Heart Disease Risk Prediction API

FastAPI microservice that predicts heart disease risk from 13 clinical inputs
using a Gradient Boosting classifier.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Builds `heart_disease_data.csv` (2000 synthetic patients, realistic clinical correlations) |
| `train_model.py` | Trains the model, prints accuracy/ROC-AUC, saves `model.pkl` |
| `main.py` | FastAPI app: Pydantic validation, OOD guardrails, `/predict` endpoint |
| `requirements.txt` | Dependencies |
| `model.pkl` | Trained model (~200 KB — safe to commit, no size issues) |

## Setup

```bash
# from inside this project folder
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
uvicorn main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

## Model performance (on held-out 20% test set)

- Accuracy: 0.89
- ROC-AUC: 0.97

These numbers are on synthetic data — treat them as a sanity check that the
pipeline works end to end, not as a claim about real-world clinical accuracy.

## Endpoints

- `GET /health`  liveness check
- `POST /predict`  takes 13 clinical fields, returns prediction + probability + OOD flag

Example request body:
```json
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
  "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
  "slope": 0, "ca": 0, "thal": 1
}
```

## OOD guardrails

Two layers:

1. **Pydantic hard bounds** (e.g. `trestbps` between 60–250) physiologically
   impossible values are rejected with `422` before they even reach the
   prediction logic.
2. **`TYPICAL_RANGES` in `main.py`** a tighter check against the range
   actually seen in training data. A value can pass Pydantic (e.g.
   `chol: 650` is physiologically plausible) but still be far outside what
   the model was trained on. These are rejected with `400`, since a
   prediction on that input would be an extrapolation, not something the
   model can be trusted on.

## Testing checklist (Swagger + Postman)

1. `GET /health` → `model_loaded: true`
2. `POST /predict` with the example above → `200`, returns `prediction`, `probability`, `confidence`
3. Send `chol: 650` → `400`, with a `violations` list naming the field and expected range
4. Send `sex: "male"` (wrong type) → `422`
5. Send a request missing a field → `422` listing exactly which field is missing
6. Send genuinely broken JSON (e.g. a trailing comma or unclosed brace) → `422`, not `500`

## Pushing to GitHub

1. Commands:
   ```bash
   git add .
   git commit -m "Heart disease risk prediction microservice"
   git push
   ```
