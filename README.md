# 🔍 Fake Job Detection System

> **DistilBERT + Heuristic Rules** — detect fraudulent job postings with NLP, Deep Learning, and explainability.

---

## 📋 Overview

A production-ready AI system that classifies job postings as **Real** or **Fake** using:

- 🤖 **DistilBERT** fine-tuned on the EMSCAD dataset
- 📏 **Heuristic rule engine** (suspicious keywords, salary checks, payment requests)
- 🔀 **Hybrid scoring**: `0.7 × BERT + 0.3 × Heuristic`
- 🧠 **SHAP + LIME** for explainability
- ⚡ **FastAPI** REST backend
- 🖥️ **Streamlit** interactive dashboard

---

## 🏗️ Project Structure

```
fake-job-detector/
├── backend/
│   ├── main.py               # FastAPI app & endpoints
│   └── report_generator.py   # PDF/CSV report generation
├── frontend/
│   └── app.py                # Streamlit dashboard
├── models/
│   ├── train_model.py        # DistilBERT training pipeline
│   ├── predictor.py          # Inference + hybrid scoring
│   └── distilbert_fake_job/  # Saved model (after training)
├── dataset/
│   ├── prepare_data.py       # Data loading + preprocessing
│   ├── train.csv             # (generated after prepare)
│   ├── val.csv
│   └── test.csv
├── explainability/
│   └── explainer.py          # SHAP + LIME wrappers
├── notebooks/
│   └── training_demo.ipynb   # Walkthrough notebook
├── uploads/                  # Temp CSV uploads
├── reports/                  # Generated reports
├── config.py                 # Central configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <repo>
cd fake-job-detector
pip install -r requirements.txt
```

### 2. (Optional) Download the real dataset

From [Kaggle EMSCAD](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction), download `fake_job_postings.csv` and place it in `dataset/`.

Without it, the system auto-generates a synthetic 2,000-row dataset.

### 3. Train the model

```bash
python models/train_model.py
```

This will:
- Prepare and split the dataset
- Fine-tune `distilbert-base-uncased` for 3 epochs
- Evaluate and save metrics to `models/test_metrics.json`
- Save the model to `models/distilbert_fake_job/`

> **GPU**: detected automatically. Training takes ~5 min on GPU, ~30 min on CPU.

### 4. Start the API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

### 5. Start the Streamlit UI

```bash
streamlit run frontend/app.py
```

Dashboard: http://localhost:8501

---

## 🐳 Docker

```bash
# Build & run everything
docker-compose up --build

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## 🔌 API Reference

### `GET /health`
```json
{ "status": "ok", "timestamp": "2024-01-01T00:00:00", "model": "distilbert-base-uncased (fine-tuned)" }
```

### `POST /predict`
```json
// Request
{ "text": "URGENT HIRING!!! Earn $5000/week...", "include_explanation": false }

// Response
{
  "prediction": "Fake",
  "scam_probability": 0.91,
  "bert_score": 0.88,
  "heuristic_score": 0.97,
  "keywords": ["registration fee", "urgent hiring"],
  "signals": [{"rule": "upfront_payment", "weight": 0.25}],
  "explanation": "This posting requests upfront payment and promises unrealistic salary.",
  "token_attributions": [{"token": "fee", "score": 0.42}, ...]
}
```

### `POST /bulk-predict`
Upload a CSV with a `text` or `description` column. Returns:
```json
{ "total": 100, "fake_count": 23, "real_count": 77, "avg_score": 0.31, "job_id": "abc12345" }
```

### `GET /report/{job_id}`
Download the bulk results CSV.

### `GET /metrics`
Returns training evaluation metrics.

---

## 🧠 Model Details

| Setting | Value |
|---------|-------|
| Base model | `distilbert-base-uncased` |
| Task | Binary classification (Real=0, Fake=1) |
| Max tokens | 256 |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Scheduler | Linear warmup |
| Early stopping | patience=2 (on F1) |

---

## 📊 Hybrid Scoring

```
final_score = 0.7 × DistilBERT_score + 0.3 × heuristic_score
```

**Heuristic signals:**

| Rule | Weight |
|------|--------|
| Suspicious keywords (per match) | 0.12 each (max 0.6) |
| Unrealistic salary | 0.20 |
| Missing company info | 0.10 |
| Upfront payment request | 0.25 |
| Excitement spam (caps/!!!) | 0.10 |

---

## 🔑 Suspicious Keywords

> registration fee · urgent hiring · no interview · work from home earn · easy money · guaranteed income · no experience needed · upfront payment · pay to apply · wire transfer · western union · money order · limited time offer · act now · immediate start · uncapped earnings · passive income · be your own boss · financial freedom · investment required · training fee · processing fee · background check fee · application fee · send money · pay first · whatsapp only · telegram only · no resume needed · same day pay · instant approval

---

## 📈 Explainability

### LIME
Perturbs the input text and measures prediction changes to identify the most influential words.

### SHAP
Uses Shapley values (Partition masker) to compute each token's contribution to the Fake probability.

### Gradient-based attribution
Backpropagates the Fake logit through the embedding layer to score token importance.

---

## 🛠️ Configuration

All settings are in `config.py`:

```python
BERT_WEIGHT       = 0.7    # weight for DistilBERT score
HEURISTIC_WEIGHT  = 0.3    # weight for rule-based score
MAX_TOKEN_LENGTH  = 256
NUM_EPOCHS        = 3
LEARNING_RATE     = 2e-5
```

---

## 📄 License

MIT License. Dataset: EMSCAD (CC BY-SA 4.0).
