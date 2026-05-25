import os
import sys
import uuid
import json
import io
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
from fastapi import FastAPI,File,UploadFile,HTTPException,BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from models.predictor import FakeJobPredictor
from config import REPORT_DIR,MODEL_DIR

app=FastAPI(
    title="Fake Job Detection API",
    description="Detect fraudulent job postings using DistilBERT and heuristics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_predictor:Optional[FakeJobPredictor]=None

def get_predictor()->FakeJobPredictor:
    global _predictor

    if _predictor is None:
        logger.info("Loading predictor")
        _predictor=FakeJobPredictor()

    return _predictor

class PredictRequest(BaseModel):
    text:str=Field(
        ...,
        min_length=10,
        example=(
            "Urgent hiring! Work from home earn $5000/week. "
            "No experience needed. Registration fee $50."
        )
    )

    include_explanation:bool=Field(
        False,
        description="Run LIME or SHAP explanations"
    )

class PredictResponse(BaseModel):
    prediction:str
    scam_probability:float
    bert_score:float
    heuristic_score:float
    keywords:list[str]
    signals:list[dict]
    explanation:str
    token_attributions:list[dict]

class BulkSummary(BaseModel):
    total:int
    fake_count:int
    real_count:int
    avg_score:float
    job_id:str

@app.get("/health",tags=["System"])
def health():
    return {
        "status":"ok",
        "timestamp":datetime.utcnow().isoformat(),
        "model":"distilbert-base-uncased"
    }

@app.get("/metrics",tags=["System"])
def get_metrics():
    metrics_path=MODEL_DIR/"test_metrics.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)

    return {
        "message":"Model not trained yet"
    }

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"]
)
def predict(req:PredictRequest):
    predictor=get_predictor()

    result=predictor.predict(req.text)

    if req.include_explanation:
        try:
            from explainability.explainer import JobExplainer

            explainer=JobExplainer(predictor)

            exp_result=explainer.explain(req.text)

            result["lime_explanation"]=exp_result.get("lime",{})
            result["shap_explanation"]=exp_result.get("shap",{})

        except Exception as e:
            logger.warning(f"Explanation failed: {e}")

    return result

@app.post(
    "/bulk-predict",
    response_model=BulkSummary,
    tags=["Prediction"]
)
async def bulk_predict(
    file:UploadFile=File(
        ...,
        description="CSV file with text or description column"
    ),
    background_tasks:BackgroundTasks=BackgroundTasks()
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            400,
            "Only CSV files are allowed"
        )

    contents=await file.read()

    try:
        df=pd.read_csv(io.BytesIO(contents))

    except Exception as e:
        raise HTTPException(
            400,
            f"Could not parse CSV: {e}"
        )

    text_col=None

    for candidate in [
        "text",
        "description",
        "job_description",
        "content"
    ]:
        if candidate in df.columns:
            text_col=candidate
            break

    if text_col is None:
        raise HTTPException(
            400,
            f"Missing text column. Found: {list(df.columns)}"
        )

    predictor=get_predictor()

    texts=df[text_col].fillna("").tolist()

    logger.info(f"Running bulk prediction on {len(texts)} rows")

    results=predictor.predict_bulk(texts)

    df["prediction"]=[
        r["prediction"] for r in results
    ]

    df["scam_probability"]=[
        r["scam_probability"] for r in results
    ]

    df["bert_score"]=[
        r["bert_score"] for r in results
    ]

    df["heuristic_score"]=[
        r["heuristic_score"] for r in results
    ]

    df["keywords"]=[
        ", ".join(r["keywords"]) for r in results
    ]

    df["explanation"]=[
        r["explanation"] for r in results
    ]

    job_id=str(uuid.uuid4())[:8]

    out_path=REPORT_DIR/f"bulk_{job_id}.csv"

    df.to_csv(out_path,index=False)

    logger.info(f"Saved bulk report to {out_path}")

    fake_count=sum(
        1 for r in results
        if r["prediction"]=="Fake"
    )

    avg_score=sum(
        r["scam_probability"] for r in results
    )/max(len(results),1)

    return {
        "total":len(results),
        "fake_count":fake_count,
        "real_count":len(results)-fake_count,
        "avg_score":round(avg_score,4),
        "job_id":job_id
    }

@app.get("/report/{job_id}",tags=["Reports"])
def download_report(job_id:str):
    report_path=REPORT_DIR/f"bulk_{job_id}.csv"

    if not report_path.exists():
        raise HTTPException(
            404,
            f"Report {job_id} not found"
        )

    return FileResponse(
        str(report_path),
        media_type="text/csv",
        filename=f"fake_job_report_{job_id}.csv"
    )

if __name__=="__main__":
    import uvicorn
    from config import API_HOST,API_PORT

    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
