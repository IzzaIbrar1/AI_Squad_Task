import pandas as pd
import numpy as np

np.random.seed(42)
n = 100000

df = pd.DataFrame({
    'User_ID': range(1, n + 1),
    'Age': np.random.normal(30, 10, n),
    'Score': np.random.normal(50, 15, n)
})

# Inject NaN values (~5%)
nan_indices_age = np.random.choice(df.index, size=5000, replace=False)
nan_indices_score = np.random.choice(df.index, size=5000, replace=False)
df.loc[nan_indices_age, 'Age'] = np.nan
df.loc[nan_indices_score, 'Score'] = np.nan

# Inject outliers
df.loc[np.random.choice(df.index, 200), 'Age'] = np.random.choice([150, -10, 200], 200)
df.loc[np.random.choice(df.index, 200), 'Score'] = np.random.choice([500, -100, 999], 200)

df.to_csv('dirty_data.csv', index=False)
print("Dataset created:", df.shape)
print(df.isnull().sum())