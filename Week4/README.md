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

---

## Day 2: Mini-Batch Training with DataLoaders & Adam Optimizer

### Upgrades from Day 14
- Wrapped tensors into TensorDataset, split into train/val (3200/800)
- Created DataLoaders: train_loader (batch_size=64, shuffle=True), val_loader (unshuffled)
- Switched optimizer to Adam (lr=0.001) from default
- Refactored training loop to iterate mini-batches (50 batches/epoch instead of 1 full-batch pass)
- Added validation evaluation每 epoch using torch.no_grad()

### Results
- Trained 25 epochs: Train Loss 0.6919 → 0.4027, Val Loss 0.5874 → 0.3849
- Best Val Loss: 0.3810 (epoch 13) early overfitting signature visible after this point
- Test Accuracy: 0.8310 (Class 1 precision 0.20, recall 0.01 still effectively non-functional for minority class)

### Files
| File | Description |
|------|-------------|
| `train_val_loss.png` | Training vs validation loss curve over 25 epochs |
| `mlp_minibatch_model.pth` | Saved model weights |