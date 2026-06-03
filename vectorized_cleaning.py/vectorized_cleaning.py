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
print("\nCleaned Data Sample:")
print(df.head())

df.to_csv('cleaned_data.csv', index=False)
print("\nCleaned data saved to cleaned_data.csv")