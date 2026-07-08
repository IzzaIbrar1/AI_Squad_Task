"""
Trains a Gradient Boosting classifier on the synthetic heart disease dataset
and serializes it to model.pkl along with the feature column order.
"""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

df = pd.read_csv("heart_disease_data.csv")

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

X = df[FEATURE_COLUMNS]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# Save model + feature order together so the API can validate at inference time
joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, "model.pkl")
print("Saved model.pkl")

import os
size_kb = os.path.getsize("model.pkl") / 1024
print(f"model.pkl size: {size_kb:.1f} KB")
