import pandas as pd
import numpy as np
import time

df = pd.read_csv('dirty_data.csv')

df_loop = df.copy()
start = time.time()

for i in range(len(df_loop)):
    if pd.isnull(df_loop.loc[i, 'Age']):
        df_loop.loc[i, 'Age'] = 30.0
    if pd.isnull(df_loop.loc[i, 'Score']):
        df_loop.loc[i, 'Score'] = 50.0

loop_time = time.time() - start
print(f"Loop time:       {loop_time:.4f} seconds")

df_vec = df.copy()
start = time.time()

df_vec['Age'] = df_vec['Age'].fillna(df_vec['Age'].median())
df_vec['Score'] = df_vec['Score'].fillna(df_vec['Score'].median())

vec_time = time.time() - start
print(f"Vectorized time: {vec_time:.4f} seconds")

print(f"\nSpeedup: {loop_time / vec_time:.1f}x faster")