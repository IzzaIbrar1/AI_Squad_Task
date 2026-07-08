"""
Generates a synthetic but clinically realistic heart disease dataset.
Feature set follows the standard Cleveland Heart Disease structure
(the most widely used schema in clinical ML), with values sampled from
ranges/correlations that mimic real patient data instead of pure noise.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000


def generate_row(has_disease):
    # Age: disease more common in older patients
    age = int(np.clip(np.random.normal(58 if has_disease else 48, 9), 29, 77))

    sex = np.random.choice([0, 1], p=[0.32, 0.68])  # 1 = male, more male cases in dataset

    # Chest pain type: 0=typical angina,1=atypical,2=non-anginal,3=asymptomatic
    if has_disease:
        cp = np.random.choice([0, 1, 2, 3], p=[0.08, 0.15, 0.22, 0.55])
    else:
        cp = np.random.choice([0, 1, 2, 3], p=[0.10, 0.30, 0.40, 0.20])

    trestbps = int(np.clip(np.random.normal(134 if has_disease else 128, 17), 94, 200))
    chol = int(np.clip(np.random.normal(250 if has_disease else 232, 45), 126, 564))
    fbs = np.random.choice([0, 1], p=[0.85, 0.15])  # fasting blood sugar > 120

    restecg = np.random.choice([0, 1, 2], p=[0.5, 0.45, 0.05])

    thalach_mean = 139 if has_disease else 158
    thalach = int(np.clip(np.random.normal(thalach_mean, 22), 71, 202))

    exang = np.random.choice([0, 1], p=[0.35, 0.65]) if has_disease else np.random.choice([0, 1], p=[0.85, 0.15])

    oldpeak = float(np.clip(np.random.exponential(1.4 if has_disease else 0.5), 0, 6.2))
    oldpeak = round(oldpeak, 1)

    slope = np.random.choice([0, 1, 2], p=[0.45, 0.45, 0.10]) if has_disease else np.random.choice([0, 1, 2], p=[0.15, 0.60, 0.25])

    ca = np.random.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.20, 0.15]) if has_disease else np.random.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.07, 0.03])

    thal = np.random.choice([1, 2, 3], p=[0.10, 0.25, 0.65]) if has_disease else np.random.choice([1, 2, 3], p=[0.05, 0.65, 0.30])

    return {
        "age": age,
        "sex": int(sex),
        "cp": int(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": int(fbs),
        "restecg": int(restecg),
        "thalach": thalach,
        "exang": int(exang),
        "oldpeak": oldpeak,
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal),
        "target": int(has_disease),
    }


rows = []
n_disease = N // 2
n_healthy = N - n_disease

for _ in range(n_disease):
    rows.append(generate_row(True))
for _ in range(n_healthy):
    rows.append(generate_row(False))

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("heart_disease_data.csv", index=False)
print(f"Generated {len(df)} rows -> heart_disease_data.csv")
print(df["target"].value_counts())
print(df.describe())
