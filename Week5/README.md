# NLP Sentiment Analysis with Hugging Face Transformers

### Objective
Clean 1,000 raw, messy customer product reviews using Regex, then classify each review's sentiment using a pre-trained Hugging Face model, no training or fine-tuning performed.

### Dataset
1,000 synthetic customer product reviews containing HTML tags, emojis, casual text, and inconsistent casing, designed to require Regex cleaning before NLP inference.

### Pipeline
1. **Text Cleaning** Regex removal of HTML tags, non-ASCII characters (emojis), and extra whitespace
2. **Pre-trained Model** `distilbert-base-uncased-finetuned-sst-2-english` via Hugging Face `pipeline()`
3. **Inference** Ran on all 1,000 cleaned reviews (CPU, no training)
4. **Confidence Thresholding** Predictions below 0.65 confidence mapped to Neutral (binary model has no native Neutral class)
5. **Export** Final CSV with `Predicted_Sentiment` and `Confidence_Score` columns

### Results
| Sentiment | Count |
|-----------|-------|
| Positive | 532 |
| Negative | 409 |
| Neutral | 59 |

### Known Limitation
Since the base model only outputs POSITIVE/NEGATIVE, genuinely neutral reviews sometimes get classified as Positive with very high confidence (e.g. "Average product, met basic expectations" → Positive, 0.9831), which the confidence threshold cannot catch. This reflects a real constraint of using a binary pre-trained model for 3-class sentiment without fine-tuning.

### Files
| File | Description |
|------|--------------|
| `nlp_sentiment_wrapper.ipynb` | Full pipeline notebook |
| `customer_reviews_raw.csv` | Source dataset |
| `customer_reviews_with_sentiment.csv` | Final exported dataset |
| `sentiment_distribution.png` | Countplot of predicted sentiment |
| `requirements.txt` | Python dependencies |

### How to Run
```bash
pip install -r requirements.txt
python -m notebook nlp_sentiment_wrapper.ipynb
```