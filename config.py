import os
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
MODEL_DIR=BASE_DIR/"models"
DATASET_DIR=BASE_DIR/"dataset"
UPLOAD_DIR=BASE_DIR/"uploads"
REPORT_DIR=BASE_DIR/"reports"
EXPLAINABILITY_DIR=BASE_DIR/"explainability"

for d in [MODEL_DIR,DATASET_DIR,UPLOAD_DIR,REPORT_DIR,EXPLAINABILITY_DIR]:
    d.mkdir(parents=True,exist_ok=True)

PRETRAINED_MODEL="distilbert-base-uncased"
SAVED_MODEL_PATH=str(MODEL_DIR/"distilbert_fake_job")
MAX_TOKEN_LENGTH=256
NUM_LABELS=2
LABEL_NAMES={0:"Real",1:"Fake"}

TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=32
NUM_EPOCHS=3
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
SEED=42
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1
TEST_SPLIT=0.1

BERT_WEIGHT=0.7
HEURISTIC_WEIGHT=0.3

SUSPICIOUS_KEYWORDS=[
    "registration fee","urgent hiring","no interview",
    "work from home earn","easy money","guaranteed income",
    "no experience needed","upfront payment","pay to apply",
    "wire transfer","western union","money order",
    "limited time offer","act now","immediate start",
    "uncapped earnings","passive income","be your own boss",
    "financial freedom","investment required","training fee",
    "processing fee","background check fee","application fee",
    "send money","pay first","whatsapp only","telegram only",
    "no resume needed","same day pay","instant approval"
]

HIGH_SALARY_THRESHOLD=50000
LOW_SALARY_THRESHOLD=1

API_HOST=os.getenv("API_HOST","0.0.0.0")
API_PORT=int(os.getenv("API_PORT",8000))

STREAMLIT_PORT=int(os.getenv("STREAMLIT_PORT",8501))
