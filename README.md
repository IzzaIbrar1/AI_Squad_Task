# AI Squad Internship Tasks — Prosensia

## Day 1: Data Ingestion Pipeline
- Generated a mock dataset of 100 rows (User ID, Age, Score)
- Saved to CSV and read back top 5 rows
- **File:** `mock_dataset.py`, `mock_users.csv`

---

## Day 2: Vectorized Data Cleaning & Benchmark Audit

### What Was Done
- Generated a dirty dataset of 100,000 rows with missing values and outliers
- Cleaned the data using fully vectorized Pandas operations (no for-loops allowed)
- Ran a benchmark comparing iterative loop vs vectorized approach
- Documented findings in a full audit report with charts

### Files
- `generate_data.py` — generates 100k row dataset with NaN and outliers
- `vectorized_cleaning.py` — cleans data using vectorized Pandas
- `benchmark.py` — execution time comparison: loop vs vectorized
- `dirty_data.csv` — raw uncleaned dataset
- `cleaned_data.csv` — final cleaned dataset
- `Benchmark Audit.docx` — full report with charts and statistical justification

### Benchmark Results
| Method | Execution Time | Speedup |
|--------|---------------|---------|
| Iterative For-Loop | 5.3384 seconds | 1x baseline |
| Vectorized Pandas | 0.0063 seconds | 851.7x faster |

### Outlier Detection
Used IQR method over Z-Score because:
- Dataset contained extreme injected outliers
- IQR is robust and makes no assumption about data distribution
- Z-Score mean and std are sensitive to extreme values

### Cleaning Results
- Original rows: 100,000
- After NaN fill + outlier removal: 97,564 rows
- Removed: 2,436 outlier records