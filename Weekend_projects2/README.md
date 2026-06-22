# Loan Default Prediction — Weekend Project
## AI/ML Internship

### Objective
Build an end-to-end classification pipeline to predict loan defaults. Clean the data, handle class imbalance, train three different algorithms, and compare them using ROC-AUC curves.

### Dataset
5,000 loan applications with 14 features (age, income, credit score, debt-to-income ratio, employment history, loan purpose, etc.) and a binary Default target. Default rate: 16.32% (genuine class imbalance).

### Project Structure
| File | Description |
|------|-------------|
| `loan_default_pipeline.ipynb` | Full classification pipeline notebook |
| `loan_default_dataset.csv` | Source dataset |
| `requirements.txt` | Python dependencies |
| `AI_Utilization_Report.md` | LLM usage documentation |
| `roc_auc_comparison.png` | ROC curve comparison chart |
| `confusion_matrices_comparison.png` | Confusion matrices for all 3 models |
| `final_loan_default_model.pkl` | Best performing model (Gradient Boosting) |
| `feature_scaler.pkl` | Fitted StandardScaler |

### Pipeline Steps
1. Data cleaning. missing value imputation, outlier capping
2. One-Hot Encoding (drop_first=True)
3. Train-Test Split (80/20, stratified, random_state=42)
4. SMOTE applied strictly to training data
5. StandardScaler fit on training data only
6. Trained Logistic Regression, Random Forest, Gradient Boosting
7. ROC-AUC comparison and final model selection

### Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.743 | 0.187 | 0.172 | 0.179 | 0.5587 |
| Random Forest | 0.800 | 0.197 | 0.074 | 0.107 | 0.5855 |
| Gradient Boosting | 0.780 | 0.194 | 0.110 | 0.141 | **0.5858** |

**Final Model:** Gradient Boosting Classifier (highest ROC-AUC)

### How to Run
```bash
pip install -r requirements.txt
jupyter notebook loan_default_pipeline.ipynb
```