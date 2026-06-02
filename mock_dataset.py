import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate mock dataset
data = {
    "User_ID": range(1, 101),
    "Age": np.random.randint(18, 60, 100),
    "Score": np.random.randint(0, 101, 100)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("mock_users.csv", index=False)

print("Dataset saved as mock_users.csv")

# Read CSV
loaded_df = pd.read_csv("mock_users.csv")

# Display top 5 rows
print("\nTop 5 Rows:")
print(loaded_df.head())