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

WEIGH_DISTILBERT=0.4
WEIGHT_PHISHING=0.2
WEIGHT_RECRUITER=0.15
WEIGHT_COMPANY=0.15
WEIGHT_SALARY=0.1

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

SUSPICIOUS_TLDS=[".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click", ".link", ".work", ".loan", ".win", ".racing", ".download", ".stream", ".gdn", "bid", ".trade"]

URL_SHORTENERS=["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly","is.gd", "buff.ly", "adf.ly", "tiny.cc", "shorte.st","clck.ru", "cutt.ly", "rb.gy", "shorturl.at", "v.gd",]

PHISHING_KEYWORDS_URL=["login", "verify", "account", "update", "secure", "bank", "password", "confirm", "click", "urgent", "suspend", "access", "security", "alert", "notice"]

LEGIT_DOMAINS={"linkedin.com", "indeed.com", "glassdoor.com", "naukri.com", "monster.com", "ziprecruiter.com", "dice.com", "lever.co","greenhouse.io", "workday.com", "taleo.net", "icims.com","smartrecruiters.com", "careers.google.com", "jobs.apple.com"}

FREE_EMAIL_PROVIDERS={"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "mail.com", "protonmail.com", "icloud.com", "ymail.com", "live.com", "msn.com", "rediffmail.com", "mailinator.com", "guerrillamail.com", "tempmail.com","throwam.com", "sharklasers.com", "yopmail.com",}

SCAM_RECRUITER_PATTERNS = [
    r"hr\s*manager\s*\d+",
    r"recruitment\s*agent\s*\d+",
    r"(mr|mrs|ms)\.?\s+[a-z]+\s+hr",
    r"whatsapp.*hiring",
    r"telegram.*jobs",
    r"apply. *whatsapp",
]

API_HOST=os.getenv("API_HOST","0.0.0.0")
API_PORT=int(os.getenv("API_PORT",8000))

STREAMLIT_PORT=int(os.getenv("STREAMLIT_PORT",8501))
