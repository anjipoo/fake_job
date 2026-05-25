"""
backend/report_generator.py
=============================
Generates PDF and CSV reports for single and bulk predictions.
"""

import sys
import io
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from config import REPORT_DIR


def generate_pdf_report(result:dict,job_text:str)->bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        logger.error("fpdf2 not installed. Run: pip install fpdf2")
        return b""

    pdf=FPDF()
    pdf.set_auto_page_break(auto=True,margin=15)
    pdf.add_page()

    pdf.set_fill_color(30,30,46)
    pdf.rect(0,0,210,30,"F")
    pdf.set_font("Helvetica","B",18)
    pdf.set_text_color(255,255,255)
    pdf.cell(0,15,"",ln=True)
    pdf.cell(0,10,"  Fake Job Detection Report",ln=True)
    pdf.set_text_color(0,0,0)
    pdf.ln(10)

    pdf.set_font("Helvetica","",9)
    pdf.set_text_color(120,120,120)
    pdf.cell(0,5,f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",ln=True)
    pdf.ln(4)

    is_fake=result["prediction"]=="Fake"
    r,g,b=(231,76,60) if is_fake else (39,174,96)

    pdf.set_fill_color(r,g,b)
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",14)

    verdict=f"  VERDICT: {result['prediction'].upper()}  |  Scam Probability: {result['scam_probability']:.1%}"

    pdf.cell(0,12,verdict,ln=True,fill=True)
    pdf.set_text_color(0,0,0)
    pdf.ln(6)

    pdf.set_font("Helvetica","B",11)
    pdf.cell(0,8,"Score Breakdown",ln=True)

    pdf.set_font("Helvetica","",10)

    scores=[
        ("Hybrid Scam Score",result["scam_probability"]),
        ("DistilBERT Score",result["bert_score"]),
        ("Heuristic Score",result["heuristic_score"]),
    ]

    for label,val in scores:
        bar_width=int(val*120)

        pdf.cell(60,6,f"{label}:",border=0)

        pdf.set_fill_color(231,76,60) if val>=0.5 else pdf.set_fill_color(39,174,96)

        pdf.cell(bar_width,5,"",fill=True)
        pdf.cell(0,5,f"  {val:.4f}",ln=True)

    pdf.ln(4)

    if result.get("keywords"):
        pdf.set_font("Helvetica","B",11)
        pdf.cell(0,8,"Suspicious Keywords Detected",ln=True)

        pdf.set_font("Helvetica","",10)

        for kw in result["keywords"]:
            pdf.set_text_color(180,0,0)
            pdf.cell(0,6,f"  \u2022 {kw}",ln=True)

        pdf.set_text_color(0,0,0)
        pdf.ln(4)

    if result.get("signals"):
        pdf.set_font("Helvetica","B",11)
        pdf.cell(0,8,"Heuristic Rule Violations",ln=True)

        pdf.set_font("Helvetica","",9)

        for sig in result["signals"]:
            rule=sig.get("rule","unknown").replace("_"," ").title()
            pdf.cell(0,5,f"  [{rule}] weight={sig.get('weight',0):.2f}",ln=True)

        pdf.ln(4)

    pdf.set_font("Helvetica","B",11)
    pdf.cell(0,8,"AI Explanation",ln=True)

    pdf.set_font("Helvetica","",10)
    pdf.multi_cell(0,6,result.get("explanation",""))
    pdf.ln(4)

    pdf.set_font("Helvetica","B",11)
    pdf.cell(0,8,"Job Posting (first 600 chars)",ln=True)

    pdf.set_font("Courier","",8)
    pdf.set_fill_color(245,245,245)

    preview=job_text[:600].replace("\n"," ")

    pdf.multi_cell(0,5,preview,fill=True)

    return bytes(pdf.output())


def generate_csv_report(results_df:pd.DataFrame)->bytes:
    buf=io.BytesIO()
    results_df.to_csv(buf,index=False)
    return buf.getvalue()


if __name__=="__main__":
    dummy_result={
        "prediction":"Fake",
        "scam_probability":0.91,
        "bert_score":0.88,
        "heuristic_score":0.97,
        "keywords":["registration fee","urgent hiring"],
        "signals":[{"rule":"upfront_payment","weight":0.25}],
        "explanation":"This posting requests upfront payment and promises unrealistic salary.",
    }

    dummy_text="URGENT HIRING!!! Earn $5000/week from home! Registration fee $99."

    pdf_bytes=generate_pdf_report(dummy_result,dummy_text)

    out=REPORT_DIR/"test_report.pdf"

    with open(out,"wb") as f:
        f.write(pdf_bytes)

    print(f"PDF saved → {out}")