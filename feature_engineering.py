import pandas as pd
import numpy as np
import holidays

# ── Step 1: Generate messy dataset ────────────────────────
np.random.seed(42)

dates_fmt1 = pd.date_range('2023-01-01', periods=500, freq='D').strftime('%Y-%m-%d %H:%M:%S')
dates_fmt2 = pd.date_range('2023-01-01', periods=500, freq='D').strftime('%d/%m/%Y')

df1 = pd.DataFrame({
    'raw_date': dates_fmt1,
    'gender': np.random.choice(['Male', 'Female', 'Unknown'], 500),
    'city': np.random.choice(['New York', 'London', 'Paris', 'Tokyo'], 500),
    'score': np.random.randint(1, 100, 500)
})

df2 = pd.DataFrame({
    'raw_date': dates_fmt2,
    'gender': np.random.choice(['Male', 'Female', 'Unknown'], 500),
    'city': np.random.choice(['New York', 'London', 'Paris', 'Tokyo'], 500),
    'score': np.random.randint(1, 100, 500)
})

df = pd.concat([df1, df2], ignore_index=True)

# Inject NaN dates
df.loc[np.random.choice(df.index, 20), 'raw_date'] = np.nan

print("Raw dataset shape:", df.shape)
print(df.head())

# ── Step 2: Parse mixed datetime formats (vectorized) ─────
mask_fmt1 = df['raw_date'].str.contains('-', na=False)
mask_fmt2 = df['raw_date'].str.contains('/', na=False)

df['parsed_date'] = pd.NaT
df.loc[mask_fmt1, 'parsed_date'] = pd.to_datetime(
    df.loc[mask_fmt1, 'raw_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df.loc[mask_fmt2, 'parsed_date'] = pd.to_datetime(
    df.loc[mask_fmt2, 'raw_date'], format='%d/%m/%Y', errors='coerce')

print("\nNaN dates after parsing:", df['parsed_date'].isnull().sum())
df = df.dropna(subset=['parsed_date'])
print("Shape after dropping unparseable dates:", df.shape)

# ── Step 3: Extract datetime features (vectorized) ────────
df['day_of_week'] = df['parsed_date'].dt.dayofweek
df['month']       = df['parsed_date'].dt.month
df['year']        = df['parsed_date'].dt.year
df['day_of_year'] = df['parsed_date'].dt.dayofyear

# ── Step 4: Holiday detection (vectorized) ────────────────
us_holidays = holidays.US(years=[2023, 2024])
df['is_holiday'] = df['parsed_date'].dt.date.map(
    lambda x: 1 if x in us_holidays else 0
)

print("\nHoliday distribution:")
print(df['is_holiday'].value_counts())

# ── Step 5: Label Encoding (vectorized) ───────────────────
df['gender_label'] = df['gender'].map({'Male': 0, 'Female': 1, 'Unknown': 2})

# ── Step 6: One-Hot Encoding (vectorized) ─────────────────
df = pd.get_dummies(df, columns=['gender', 'city'], prefix=['gender', 'city'], dtype=int)

print("\nFinal dataframe shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample output:")
print(df.head())

# ── Step 7: Save ──────────────────────────────────────────
df.to_csv('feature_engineered_data.csv', index=False)
print("\nSaved to feature_engineered_data.csv")