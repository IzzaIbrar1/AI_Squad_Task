import pandas as pd
import numpy as np


df = pd.read_csv('dirty_data.csv')
print("Original shape:", df.shape)
print("Missing values before:\n", df.isnull().sum())


df['Age'] = df['Age'].fillna(df['Age'].median())
df['Score'] = df['Score'].fillna(df['Score'].median())
print("\nMissing values after fillna:\n", df.isnull().sum())


def remove_outliers_iqr(dataframe, column):
    """Remove outliers from a column using the IQR method."""
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return dataframe[(dataframe[column] >= lower) & (dataframe[column] <= upper)]


df = remove_outliers_iqr(df, 'Age')
df = remove_outliers_iqr(df, 'Score')
print("\nShape after outlier removal:", df.shape)


df_standard = df.copy()
cols = ['Age', 'Score']

df_standard[cols] = (df_standard[cols] - df_standard[cols].mean()) / df_standard[cols].std()

print("\nStandard Scaled Sample:")
print(df_standard[['Age', 'Score']].head())
print("Mean after scaling (should be ~0):", df_standard[cols].mean().round(4).to_dict())
print("Std after scaling (should be ~1):", df_standard[cols].std().round(4).to_dict())


df_minmax = df.copy()

df_minmax[cols] = (df_minmax[cols] - df_minmax[cols].min()) / (
    df_minmax[cols].max() - df_minmax[cols].min()
)

print("\nMinMax Scaled Sample:")
print(df_minmax[['Age', 'Score']].head())
print("Min after scaling (should be 0):", df_minmax[cols].min().to_dict())
print("Max after scaling (should be 1):", df_minmax[cols].max().to_dict())


df.to_csv('cleaned_data.csv', index=False)
df_standard.to_csv('standard_scaled_data.csv', index=False)
df_minmax.to_csv('minmax_scaled_data.csv', index=False)

print("\nAll files saved successfully.")
print("cleaned_data.csv —", df.shape)
print("standard_scaled_data.csv —", df_standard.shape)
print("minmax_scaled_data.csv —", df_minmax.shape)