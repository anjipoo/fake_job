import sys
import io
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="FakeJob Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS="""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.main{background:#0f1117;}
.verdict-fake{background:linear-gradient(135deg,#2d0a0a 0%,#1a0808 100%);border:1px solid #e74c3c;border-left:4px solid #e74c3c;border-radius:12px;padding:20px 24px;margin:12px 0;}
.verdict-real{background:linear-gradient(135deg,#0a2d0a 0%,#081a08 100%);border:1px solid #2ecc71;border-left:4px solid #2ecc71;border-radius:12px;padding:20px 24px;margin:12px 0;}
.verdict-title{font-size:2rem;font-weight:700;letter-spacing:0.04em;margin:0;}
.fake-color{color:#e74c3c;}
.real-color{color:#2ecc71;}
.kw-pill{display:inline-block;background:rgba(231,76,60,0.15);border:1px solid rgba(231,76,60,0.5);color:#e74c3c;border-radius:20px;padding:3px 12px;margin:3px;font-size:0.82rem;font-family:'JetBrains Mono',monospace;}
.metric-card{background:#1a1d2e;border:1px solid #2d3056;border-radius:10px;padding:16px 20px;text-align:center;}
.metric-number{font-size:2rem;font-weight:700;color:#7c83fd;font-family:'JetBrains Mono',monospace;}
.metric-label{font-size:0.8rem;color:#888;margin-top:4px;}
.signal-badge{background:rgba(255,165,0,0.12);border:1px solid rgba(255,165,0,0.4);color:#ffb300;border-radius:6px;padding:4px 10px;margin:3px;font-size:0.78rem;display:inline-block;}
</style>
"""
st.markdown(CUSTOM_CSS,unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading AI model …")
def load_predictor():
    from models.predictor import FakeJobPredictor
    return FakeJobPredictor()

@st.cache_resource(show_spinner="Loading explainability engines …")
def load_explainer():
    from explainability.explainer import JobExplainer
    return JobExplainer(load_predictor())

def gauge_chart(score:float)->go.Figure:
    color="#e74c3c" if score>=0.5 else "#2ecc71"
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=score*100,
        title={"text":"Scam Probability (%)","font":{"size":14,"color":"#aaa"}},
        number={"suffix":"%","font":{"size":28,"color":color}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#555","tickwidth":1},
            "bar":{"color":color,"thickness":0.25},
            "bgcolor":"#1a1d2e",
            "bordercolor":"#2d3056",
            "steps":[
                {"range":[0,40],"color":"rgba(46,204,113,0.15)"},
                {"range":[40,65],"color":"rgba(243,156,18,0.15)"},
                {"range":[65,100],"color":"rgba(231,76,60,0.15)"}
            ],
            "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":score*100}
        }
    ))
    fig.update_layout(height=240,margin=dict(l=20,r=20,t=30,b=0),paper_bgcolor="rgba(0,0,0,0)",font_color="#ccc")
    return fig

def score_bar(label:str,value:float)->go.Figure:
    color="#e74c3c" if value>=0.5 else "#2ecc71"
    fig=go.Figure(go.Bar(
        x=[value],y=[label],
        orientation="h",
        marker_color=color,
        text=[f"{value:.3f}"],
        textposition="outside"
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1],showgrid=False,color="#555"),
        yaxis=dict(showgrid=False,color="#ccc"),
        height=60,margin=dict(l=0,r=40,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#ccc",size=11)
    )
    return fig

def highlight_text(text:str,keywords:list[str])->str:
    from config import SUSPICIOUS_KEYWORDS
    import re
    all_kw=list(set(keywords+SUSPICIOUS_KEYWORDS))
    result=text
    for kw in sorted(all_kw,key=len,reverse=True):
        if kw.lower() in result.lower():
            pattern=re.compile(re.escape(kw),re.IGNORECASE)
            result=pattern.sub(f'<mark style="background:#7a1a1a;color:#ffb3b3;border-radius:3px;padding:1px 3px">{kw}</mark>',result)
    return result

def render_verdict_card(result:dict):
    is_fake=result["prediction"]=="Fake"
    cls="verdict-fake" if is_fake else "verdict-real"
    color="fake-color" if is_fake else "real-color"
    icon="🚨" if is_fake else "✅"
    st.markdown(f"""
    <div class="{cls}">
    <p class="verdict-title {color}">{icon}{result["prediction"].upper()}</p>
    <p style="color:#aaa;margin:4px 0 0 0;font-size:0.9rem">
    This job posting appears to be <strong class="{color}">{result["prediction"].lower()}</strong>
    with a scam probability of <strong class="{color}">{result["scam_probability"]:.1%}</strong>.
    </p>
    </div>
    """,unsafe_allow_html=True)

def render_keywords(keywords:list[str]):
    if not keywords:
        st.info("No suspicious keywords detected.")
        return
    pills="".join(f'<span class="kw-pill">⚠{kw}</span>' for kw in keywords)
    st.markdown(f'<div style="margin:8px 0">{pills}</div>',unsafe_allow_html=True)

def render_signals(signals:list[dict]):
    if not signals:
        st.success("No heuristic rule violations.")
        return
    for sig in signals:
        rule=sig.get("rule","").replace("_"," ").title()
        details=sig.get("details","")
        weight=sig.get("weight",0)
        st.markdown(f'<span class="signal-badge">🔴{rule}—{details}(w={weight:.2f})</span>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔍 FakeJob Detector")
    st.markdown("*DistilBERT+Heuristics*")
    st.divider()
    page=st.radio("Navigation",["🔎 Single Analysis","📁 Bulk Upload","📊 Analytics","ℹ️ About"],label_visibility="collapsed")
    st.divider()
    show_lime=st.toggle("LIME Explanation",value=False)
    show_shap=st.toggle("SHAP Explanation",value=False)
    show_tokens=st.toggle("Token Attributions",value=True)
    confidence_thresh=st.slider("Fake Threshold",0.3,0.8,0.5,0.05)

if "🔎 Single Analysis"in page:
    st.markdown("## 🔎 Analyse Job")
    col1,col2,col3=st.columns(3)
    if col1.button("Fake"):
        st.session_state["sample_text"]="URGENT HIRING!!! Earn $5000/week!!!"
    if col2.button("Real"):
        st.session_state["sample_text"]="Hiring Software Engineer with 5 years experience"
    if col3.button("Clear"):
        st.session_state["sample_text"]=""
    job_text=st.text_area("Job",value=st.session_state.get("sample_text",""),height=180)
    analyse_btn=st.button("Analyse")
    if analyse_btn and job_text.strip():
        predictor=load_predictor()
        result=predictor.predict(job_text)
        result["prediction"]="Fake" if result["scam_probability"]>=confidence_thresh else "Real"
        st.markdown("---")
        l,r=st.columns([2,1])
        with l:
            render_verdict_card(result)
        with r:
            st.plotly_chart(gauge_chart(result["scam_probability"]),use_container_width=True)
        st.markdown("### Breakdown")
        st.metric("Hybrid",f"{result['scam_probability']:.4f}")
        st.metric("BERT",f"{result['bert_score']:.4f}")
        st.metric("Heuristic",f"{result['heuristic_score']:.4f}")
        st.markdown("### Text")
        st.markdown(highlight_text(job_text,result["keywords"]),unsafe_allow_html=True)
        st.markdown("### Keywords")
        render_keywords(result["keywords"])
        st.markdown("### Signals")
        render_signals(result["signals"])