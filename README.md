# AI Squad Internship Tasks — Prosensia

## Week 1

### Day 1: Data Ingestion Pipeline
- Generated a mock dataset of 100 rows (User ID, Age, Score)
- Saved to CSV and read back top 5 rows
- **Files:** `mock_dataset.py`, `mock_users.csv`

---

### Day 2: Vectorized Data Cleaning & Benchmark Audit
- Generated a dirty dataset of 100,000 rows with missing values and outliers
- Cleaned data using fully vectorized Pandas (no for-loops)
- Benchmarked loop vs vectorized: 851.7x speedup
- **Files:** `generate_data.py`, `vectorized_cleaning.py`, `benchmark.py`, `dirty_data.csv`, `cleaned_data.csv`, `Benchmark Audit.docx`

---

### Day 3: Normalization & Feature Scaling
- Finalized the cleaning pipeline with StandardScaler and MinMaxScaler
- Both scalers implemented using vectorized Pandas operations
- Documented mathematical differences between Standard and MinMax scaling
- Explained why wrong scaling causes severe errors in SVM models
- **Files:** `standard_scaled_data.csv`, `minmax_scaled_data.csv`, `Normalization brief.docx`

---

### Day 4: Feature Engineering Pipeline & Feature Encoding Brief
- Built a feature engineering pipeline on a messy datetime dataset
- Extracted day of week, month, year, day of year, and US holiday status
- Applied Label Encoding and One-Hot Encoding using vectorized Pandas
- Documented why raw strings break models and the difference between encoding methods
- **Files:** `feature_engineering.py`, `feature_engineered_data.csv`, `Feature Encoding Brief.docx`

---

## Week 2

### Day 5: Environment Scaffold & Vectorized Data Cleaning Pipeline
- Installed matplotlib and regenerated requirements.txt
- Refactored vectorized_cleaning.py to be fully PEP-8 compliant
- Removed all for-loops from scaling operations — fully vectorized
- Verified clean CSV output with 97,564 rows
- **Files:** `requirements.txt`, `vectorized_cleaning.py`

---

### Day 6: Feature Engineering & EDA — Jupyter Notebook
- Ingested cleaned dataset (97,564 rows) from Day 5 pipeline
- Extracted datetime features: day of week, month, weekend flag, holiday status
- Applied One-Hot Encoding on age groups using vectorized Pandas
- Generated 2 distribution plots (Age and Score) using Matplotlib and Seaborn
- Generated correlation heatmap across all engineered features
- Exported final dataset as features_v1.csv
- **Files:** `eda_feature_engineering.ipynb`, `features_v1.csv`