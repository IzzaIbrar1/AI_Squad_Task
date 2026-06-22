# Deep Learning Baseline: PyTorch MLP
## Week 3, Day 1: Deep Learning Foundations & Tensor Architecture

### Objective
Transition from Scikit-Learn to PyTorch. Convert the cleaned loan default dataset into PyTorch tensors, build a custom Multi-Layer Perceptron (MLP), and train it using a manual training loop no `.fit()` abstractions.

### Dataset
Loan default dataset (5,000 rows) from the Week 2 Weekend Project. 83.7% non-default, 16.3% default.

### Architecture
- Input layer: 22 features
- Hidden layer: 16 units with ReLU activation
- Output layer: 2 units (binary classification)

### Training
- Manual loop: zero_grad → forward pass → CrossEntropyLoss → backward() → optimizer.step()
- Optimizer: Adam, learning rate 0.01
- Epochs: 20
- Loss reduced from 0.8028 → 0.4291

### Results
- Test Accuracy: 0.8370
- Class 1 (Default) Precision/Recall: 0.00 model collapsed into majority-class prediction (Accuracy Fallacy, consistent with Week 2 findings)

### Key Files
| File | Description |
|------|-------------|
| `deep_learning_baseline.ipynb` | Full PyTorch MLP pipeline |
| `loan_default_dataset.csv` | Source dataset |
| `requirements.txt` | Python dependencies |
| `training_loss.png` | Loss curve over 20 epochs |

### How to Run
```bash
pip install -r requirements.txt
jupyter notebook deep_learning_baseline.ipynb
```