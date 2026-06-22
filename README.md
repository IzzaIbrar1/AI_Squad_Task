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

---

### Day 7: Scikit-Learn Model Training & Cross-Validation Pipeline
- Trained RandomForestClassifier and LogisticRegression on features_v1.csv
- Applied class_weight='balanced' to handle class imbalance
- Ran 5-fold StratifiedKFold cross-validation on both models
- Random Forest CV F1: 0.4795 | Logistic Regression CV F1: 0.3345
- Generated confusion matrices and classification reports
- Saved both models as .pkl files using joblib
- **Files:** `model_training.ipynb`, `random_forest_model.pkl`, `logistic_regression_model.pkl`, `confusion_matrices.png`

---

### Day 8: Feature Engineering Pipeline — Dynamic Datetime Extraction & Categorical Encoding
- Loaded final_clean_v2.csv and added datetime column
- Extracted Month, DayOfWeek, and Is_Weekend features using vectorized Pandas
- Added Department and City categorical columns
- Applied One-Hot Encoding with drop_first=True to avoid Dummy Variable Trap
- Generated distribution plots for Age, Score, and Month
- Exported model-ready dataset as model_ready_features.csv
- **Files:** `feature_pipeline.ipynb`, `final_clean_v2.csv`, `model_ready_features.csv`, `feature_distributions.png`

---

## Weekend Project: E-Commerce Sales EDA

- Performed full EDA on 10,000 e-commerce orders (2021–2024)
- Data cleaning, preprocessing and datetime parsing
- Generated correlation heatmap and 4 distribution/trend charts
- Extracted 6 statistical business insights
- **Files:** `weekend_projects/EDA_Ecommerce.ipynb`, `weekend_projects/ecommerce_cleaned.csv`

---

## Week 3

---

### Day 9: Baseline Logistic Regression Model — Train-Test Split & Evaluation
- Created binary target variable: Is_High_Score (Score > 50 = 1, else 0)
- 80/20 Train-Test Split with random_state=42
- Trained LogisticRegression (max_iter=1000) on training data only
- Model Accuracy: 53% — synthetic dataset with no real feature-target relationship
- Pipeline is correct, data quality is the limiting factor
- **Files:** `baseline_model.ipynb`, `model_ready_features.csv`, `requirements.txt`

---

### Day 10: Random Forest Classifier — Ensemble Modeling & Advanced Classification Metrics
- Trained RandomForestClassifier (n_estimators=100, random_state=42) on existing X_train/y_train split
- Evaluated using Precision, Recall, F1-Score instead of relying only on Accuracy
- Random Forest: Accuracy 0.5024 | Precision 0.5269 | Recall 0.5479 | F1 0.5372
- Logistic Regression: Accuracy 0.5271 | Precision 0.5271 | Recall 1.0000 | F1 0.6903
- Identified Logistic Regression's Recall of 1.0000 as misleading — model predicting class 1 almost universally
- Generated confusion matrix for Random Forest
- Saved trained model as random_forest_baseline.pkl
- **Files:** `baseline_model.ipynb`, `random_forest_baseline.pkl`, `rf_confusion_matrix.png`

---

### Day 11: Hyperparameter Tuning with GridSearchCV; Bias-Variance Tradeoff
- Created param_grid: n_estimators [10,50,100], max_depth [5,10,20], min_samples_split [2,5]
- Ran GridSearchCV with 5-fold cross-validation on X_train only
- Best Parameters: max_depth=5, min_samples_split=2, n_estimators=50
- Tuned RF: Accuracy 0.5270 | Precision 0.5271 | Recall 0.9985 | F1 0.6900
- Identified that high F1 score masked a class collapse, model predicted majority class almost universally (Recall 0.00 on class 0)
- Documented why optimizing for plain F1 score can hide imbalanced per-class performance
- Compared tuned model against Day 10 baseline using updated confusion matrix
- Overwrote random_forest_baseline.pkl with optimized model
- **Files:** `Week3/baseline_model.ipynb`, `Week3/random_forest_baseline.pkl`, `Week3/tuned_rf_confusion_matrix.png`

---

### Day 12: SMOTE Class Balancing & Feature Importance Interpretability
- Audited y_train class distribution: 52.5% vs 47.5% (mild, not severe imbalance)
- Applied SMOTE strictly to X_train/y_train using K-Nearest Neighbors interpolation
- Retrained tuned Random Forest on SMOTE balanced data (X_test/y_test untouched throughout)
- Post-SMOTE: Accuracy 0.5035 | Precision 0.5238 | Recall 0.6366 | F1 0.5747
- Identified pre-SMOTE F1 of 0.69 as artificially inflated by majority class collapse
- Extracted .feature_importances_ Top features: DayOfWeek, City_Paris, Age
- Documented business interpretation of feature importance chart
- Exported production_rf_model.pkl
- **Files:** `Week3/baseline_model.ipynb`, `Week3/production_rf_model.pkl`, `Week3/feature_importance.png`

---

## Weekend Project 2: Loan Default Prediction — End-to-End Classification Pipeline

- Built a complete classification pipeline on a 5,000-row loan default dataset (16.32% default rate)
- Data cleaning, One-Hot Encoding, stratified 80/20 split, SMOTE (train-only), StandardScaler (leak-free)
- Trained and compared 3 algorithms: Logistic Regression, Random Forest, Gradient Boosting
- Selected

---

### Day 14: PyTorch MLP: Deep Learning Foundations & Manual Training Loop
- Converted loan default dataset into PyTorch FloatTensor/LongTensor
- Built custom MLP (22 → 16 ReLU → 2) using torch.nn.Module
- Implemented manual training loop (zero_grad, forward, loss, backward, optimizer.step) no .fit() used
- Trained 20 epochs: loss reduced from 0.8028 to 0.4291
- Test Accuracy: 0.8370, but model collapsed into majority-class prediction (0.00 precision/recall on minority class)  same Accuracy Fallacy pattern as Week 2
- Documented difference between NumPy arrays and PyTorch Tensors (autograd, device flexibility)
- Explained mathematically why ReLU solves vanishing gradients vs Sigmoid
- **Files:** `Week4/deep_learning_baseline.ipynb`, `Week4/training_loss.png`