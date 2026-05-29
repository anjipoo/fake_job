# Fake Job Detection System

AI-powered Recruitment Fraud Detection Platform built using DistilBERT, Explainable AI, and Cybersecurity Intelligence.

## Overview

RecruitShield AI helps job seekers identify fraudulent job postings, suspicious recruiters, and recruitment scams.

The platform combines:

* DistilBERT fine-tuned on the EMSCAD Fake Job Posting dataset
* Rule-based fraud detection
* Hybrid risk scoring
* SHAP & LIME explainability
* FastAPI backend
* Streamlit dashboard
* Hugging Face model deployment

## Features

### Fake Job Detection

* DistilBERT-based classification
* Real vs Fake prediction
* Scam probability score

### Explainable AI

* SHAP explanations
* LIME explanations
* Token attribution analysis

### Fraud Intelligence

* Suspicious keyword detection
* Upfront payment detection
* Unrealistic salary detection
* Missing company information detection

### Bulk Analysis

* CSV upload support
* Batch prediction
* Downloadable reports

## Tech Stack

| Category       | Technology            |
| -------------- | --------------------- |
| NLP            | DistilBERT            |
| ML             | PyTorch, Transformers |
| Explainability | SHAP, LIME            |
| Backend        | FastAPI               |
| Frontend       | Streamlit             |
| Visualization  | Plotly                |
| Deployment     | Streamlit Cloud       |
| Model Hosting  | Hugging Face          |

## Project Structure

```text
fake_job/
├── backend/
├── frontend/
├── models/
├── explainability/
├── dataset/
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

EMSCAD Fake Job Postings Dataset

* 17,880 job postings
* 866 fraudulent postings
* Binary classification task

Dataset Source:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

## Model

Base Model:
distilbert-base-uncased

Hosted on Hugging Face:
https://huggingface.co/anjipoo/fake_job_detector

## Local Setup

### Clone Repository

```bash
git clone https://github.com/anjipoo/fake_job.git
cd fake_job
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run frontend/app.py
```

### Run FastAPI

```bash
uvicorn backend.main:app --reload
```

## Hybrid Scoring

```text
Final Score =
0.7 × DistilBERT Score +
0.3 × Heuristic Score
```

## Future Enhancements

* Phishing URL Detection
* Scam Recruiter Detection
* Company Trust Scoring
* Salary Anomaly Detection
* Browser Extension
* Graph-Based Fraud Intelligence

## License

MIT License
