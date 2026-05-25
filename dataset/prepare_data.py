"""
datasets/prepare_data.py
========================
Downloads and preprocesses the EMSCAD (Fake Job Postings) dataset.
Dataset: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
Falls back to a built-in synthetic dataset if Kaggle is unavailable.
"""

import os
import re
import sys
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR,TRAIN_SPLIT,VAL_SPLIT,TEST_SPLIT,SEED

random.seed(SEED)
np.random.seed(SEED)

# REAL_TEMPLATES=[
#     "We are looking for a {title} to join our team at {company}. "
#     "The ideal candidate has {exp} years of experience in {field}. "
#     "We offer competitive salary, health insurance, and 401k. "
#     "Responsibilities: {resp}. Requirements: Bachelor's degree preferred. "
#     "Apply through our website with your resume and cover letter.",

#     "{company} is hiring a {title}. This is a full-time position with "
#     "salary range ${sal_low}–${sal_high}/year. You will work with our "
#     "{field} team to {resp}. We value diversity and inclusion. "
#     "Send your CV to careers@{company_domain}.com.",

#     "Exciting opportunity for a {title} at {company}! "
#     "You'll be responsible for {resp}. "
#     "We require {exp}+ years in {field}. Benefits include remote work, "
#     "dental/vision, and paid time off. EOE.",
# ]

# FAKE_TEMPLATES=[
#     "URGENT HIRING!!! {title} needed IMMEDIATELY. No experience required! "
#     "Earn ${fake_sal}/week working from home. No interview needed. "
#     "Registration fee of $50 required to process your application. "
#     "WhatsApp us NOW: +1-555-{phone}. Limited spots available!",

#     "Work from home and earn ${fake_sal} daily! Be your own boss. "
#     "No resume needed. Immediate start. Send $99 processing fee via Western Union. "
#     "Guaranteed income, passive earnings, financial freedom. Act NOW!",

#     "MAKE ${fake_sal}/MONTH EASILY! {title} position open. "
#     "No skills needed. Upfront payment of $200 to receive your work kit. "
#     "100% guaranteed. Telegram only: @scamjob_{phone}. "
#     "Pay to apply and start earning same day!",
# ]

TITLES=["Data Analyst","Software Engineer","Marketing Manager",
         "Sales Associate","Customer Support","HR Specialist",
         "Project Manager","Graphic Designer","Content Writer",
         "Accountant","Business Analyst","DevOps Engineer"]

COMPANIES=["Acme Corp","TechSolutions","GlobalVentures","BrightPath",
           "NovaSystems","PrimeWorks","Apex Industries","CoreLogic"]

FIELDS=["technology","marketing","finance","operations",
         "design","engineering","sales","analytics"]

RESPS=[
    "develop and maintain software systems",
    "manage client relationships and grow accounts",
    "analyse business data and produce reports",
    "coordinate cross-functional projects",
    "create compelling marketing campaigns",
    "provide technical support to end-users",
]


# def _make_synthetic_dataset(n:int=2000)->pd.DataFrame:
#     rows=[]
#     half=n//2

#     for _ in range(half):
#         tmpl=random.choice(REAL_TEMPLATES)

#         sal_low=random.randint(40,90)*1000
#         sal_high=sal_low+random.randint(10,30)*1000

#         text=tmpl.format(
#             title=random.choice(TITLES),
#             company=random.choice(COMPANIES),
#             exp=random.randint(1,8),
#             field=random.choice(FIELDS),
#             resp=random.choice(RESPS),
#             sal_low=f"{sal_low:,}",
#             sal_high=f"{sal_high:,}",
#             company_domain=random.choice(COMPANIES).lower().replace(" ",""),
#         )

#         rows.append({
#             "text":text,
#             "label":0,
#             "label_name":"Real"
#         })

#     for _ in range(half):
#         tmpl=random.choice(FAKE_TEMPLATES)

#         text=tmpl.format(
#             title=random.choice(TITLES),
#             fake_sal=random.randint(500,5000),
#             phone="".join([str(random.randint(0,9)) for _ in range(7)]),
#         )

#         rows.append({
#             "text":text,
#             "label":1,
#             "label_name":"Fake"
#         })

#     df=pd.DataFrame(rows).sample(frac=1,random_state=SEED).reset_index(drop=True)

#     return df


def _load_emscad(csv_path:str)->pd.DataFrame:
    df=pd.read_csv(csv_path)

    logger.info(f"Loaded EMSCAD dataset: {len(df)} rows")

    text_cols=[
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    for col in text_cols:
        if col not in df.columns:
            df[col]=""

        df[col]=df[col].fillna("")

    df["text"]=(
        df["title"].str.strip()+" [SEP] "+
        df["company_profile"].str.strip()+" [SEP] "+
        df["description"].str.strip()+" [SEP] "+
        df["requirements"].str.strip()+" [SEP] "+
        df["benefits"].str.strip()
    )

    df["label"]=df["fraudulent"].astype(int)

    df["label_name"]=df["label"].map({
        0:"Real",
        1:"Fake"
    })

    return df[["text","label","label_name"]]


def clean_text(text:str)->str:
    if not isinstance(text,str):
        return ""

    text=re.sub(r"<[^>]+>"," ",text)
    text=re.sub(r"http\S+|www\.\S+"," ",text)
    text=re.sub(r"[^\w\s.,!?;:()\-]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()

    return text


def prepare_dataset(kaggle_csv:str|None=None)->dict[str,pd.DataFrame]:
    emscad_path=DATASET_DIR/"fake_job_postings.csv"

    if kaggle_csv and Path(kaggle_csv).exists():
        df=_load_emscad(kaggle_csv)

    elif emscad_path.exists():
        df = _load_emscad(str(emscad_path))
        logger.info("Found EMSCAD dataset.")

    else:
        raise FileNotFoundError(
            f"""
    EMSCAD dataset not found.

    Place fake_job_postings.csv inside:
    {DATASET_DIR}

    Download from:
    https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
    """
        )

    logger.info("Cleaning text …")

    df["text"]=df["text"].apply(clean_text)

    df=df[df["text"].str.len()>20].reset_index(drop=True)

    counts=df["label"].value_counts()

    logger.info(f"Class balance — Real: {counts.get(0,0)}, Fake: {counts.get(1,0)}")

    train_df,temp_df=train_test_split(
        df,
        test_size=(1-TRAIN_SPLIT),
        random_state=SEED,
        stratify=df["label"]
    )

    relative_val=VAL_SPLIT/(VAL_SPLIT+TEST_SPLIT)

    val_df,test_df=train_test_split(
        temp_df,
        test_size=(1-relative_val),
        random_state=SEED,
        stratify=temp_df["label"]
    )

    logger.info(f"Split → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    for split_name,split_df in [
        ("train",train_df),
        ("val",val_df),
        ("test",test_df)
    ]:
        out_path=DATASET_DIR/f"{split_name}.csv"

        split_df.to_csv(out_path,index=False)

        logger.info(f"Saved {split_name} → {out_path}")

    return {
        "train":train_df,
        "val":val_df,
        "test":test_df
    }


if __name__=="__main__":
    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default=None,
        help="Path to fake_job_postings.csv"
    )

    args=parser.parse_args()

    splits=prepare_dataset(args.csv)

    print("\nDataset prepared successfully!")

    for k,v in splits.items():
        print(f"  {k}: {len(v)} rows")