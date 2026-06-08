# AI Squad Internship Tasks — Prosensia

## Day 1: Data Ingestion Pipeline
- Generated a mock dataset of 100 rows (User ID, Age, Score)
- Saved to CSV and read back top 5 rows
- **Files:** `mock_dataset.py`, `mock_users.csv`

---

## Day 2: Vectorized Data Cleaning & Benchmark Audit
- Generated a dirty dataset of 100,000 rows with missing values and outliers
- Cleaned data using fully vectorized Pandas (no for-loops)
- Benchmarked loop vs vectorized: 851.7x speedup
- **Files:** `generate_data.py`, `vectorized_cleaning.py`, `benchmark.py`, `dirty_data.csv`, `cleaned_data.csv`, `Benchmark Audit.docx`

---

## Day 3: Normalization & Feature Scaling
- Finalized the cleaning pipeline with StandardScaler and MinMaxScaler
- Both scalers implemented using vectorized Pandas operations
- Documented mathematical differences between Standard and MinMax scaling
- Explained why wrong scaling causes severe errors in SVM models
- **Files:** `standard_scaled_data.csv`, `minmax_scaled_data.csv`, `Normalization brief.docx`

---

## Day 4: Feature Engineering Pipeline & Feature Encoding Brief
- Built a feature engineering pipeline on a messy datetime dataset
- Extracted day of week, month, year, day of year, and US holiday status
- Applied Label Encoding and One-Hot Encoding using vectorized Pandas
- Documented why raw strings break models and the difference between encoding methods
- **Files:** `feature_engineering.py`, `feature_engineered_data.csv`, `Feature Encoding Brief.docx`

---

## Day 5: Environment Scaffold & Vectorized Data Cleaning Pipeline
- Installed matplotlib and regenerated requirements.txt
- Refactored vectorized_cleaning.py to be fully PEP-8 compliant
- Removed all for-loops from scaling operations — fully vectorized
- Verified clean CSV output with 97,564 rows
- **Files:** `requirements.txt`, `vectorized_cleaning.py`
