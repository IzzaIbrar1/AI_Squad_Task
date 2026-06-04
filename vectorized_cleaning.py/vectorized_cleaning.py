import pandas as pd
import numpy as np

df = pd.read_csv('dirty_data.csv')
print("Original shape:", df.shape)
print("Missing values before:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Score'] = df['Score'].fillna(df['Score'].median())
print("\nMissing values after fillna:\n", df.isnull().sum())

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers_iqr(df, 'Age')
df = remove_outliers_iqr(df, 'Score')
print("\nShape after outlier removal:", df.shape)

df_standard = df.copy()
for col in ['Age', 'Score']:
    mean = df_standard[col].mean()
    std = df_standard[col].std()
    df_standard[col] = (df_standard[col] - mean) / std

print("\nStandard Scaled Sample:")
print(df_standard[['Age', 'Score']].head())
print("Mean after scaling (should be ~0):", df_standard[['Age','Score']].mean().round(4).to_dict())
print("Std after scaling (should be ~1):", df_standard[['Age','Score']].std().round(4).to_dict())

df_minmax = df.copy()
for col in ['Age', 'Score']:
    min_val = df_minmax[col].min()
    max_val = df_minmax[col].max()
    df_minmax[col] = (df_minmax[col] - min_val) / (max_val - min_val)

print("\nMinMax Scaled Sample:")
print(df_minmax[['Age', 'Score']].head())
print("Min after scaling (should be 0):", df_minmax[['Age','Score']].min().to_dict())
print("Max after scaling (should be 1):", df_minmax[['Age','Score']].max().to_dict())

df.to_csv('cleaned_data.csv', index=False)
df_standard.to_csv('standard_scaled_data.csv', index=False)
df_minmax.to_csv('minmax_scaled_data.csv', index=False)

print("\nAll files saved successfully.")
print("cleaned_data.csv —", df.shape)
print("standard_scaled_data.csv —", df_standard.shape)
print("minmax_scaled_data.csv —", df_minmax.shape)