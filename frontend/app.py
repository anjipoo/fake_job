import os
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

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ─── Page config (must be first Streamlit call) ────
st.set_page_config(
    page_title  = "FakeJob Detector",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ────────────────────────────────────
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  .main { background: #0f1117; }

  /* Verdict card */
  .verdict-fake {
    background: linear-gradient(135deg, #2d0a0a 0%, #1a0808 100%);
    border: 1px solid #e74c3c;
    border-left: 4px solid #e74c3c;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
  }
  .verdict-real {
    background: linear-gradient(135deg, #0a2d0a 0%, #081a08 100%);
    border: 1px solid #2ecc71;
    border-left: 4px solid #2ecc71;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
  }
  .verdict-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 0;
  }
  .fake-color  { color: #e74c3c; }
  .real-color  { color: #2ecc71; }
  .score-label { font-size: 0.8rem; color: #888; margin-bottom: 2px; }
  .score-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; }

  /* Keyword pill */
  .kw-pill {
    display: inline-block;
    background: rgba(231,76,60,0.15);
    border: 1px solid rgba(231,76,60,0.5);
    color: #e74c3c;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Metric card */
  .metric-card {
    background: #1a1d2e;
    border: 1px solid #2d3056;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-number {
    font-size: 2rem;
    font-weight: 700;
    color: #7c83fd;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-label { font-size: 0.8rem; color: #888; margin-top: 4px; }

  /* Signal badge */
  .signal-badge {
    background: rgba(255,165,0,0.12);
    border: 1px solid rgba(255,165,0,0.4);
    color: #ffb300;
    border-radius: 6px;
    padding: 4px 10px;
    margin: 3px;
    font-size: 0.78rem;
    display: inline-block;
  }

  /* Header */
  .app-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #12141f 100%);
    border-bottom: 1px solid #2d3056;
    padding: 20px 0;
    margin-bottom: 24px;
  }

  div[data-testid="stTextArea"] textarea {
    font-family: 'Space Grotesk', sans-serif;
    background: #1a1d2e !important;
    border: 1px solid #2d3056 !important;
    color: #e8e8e8 !important;
    border-radius: 8px;
  }

  .stButton > button {
    background: linear-gradient(135deg, #7c83fd 0%, #5c63d8 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    padding: 10px 32px;
    font-size: 0.95rem;
    transition: all 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# LAZY-LOAD PREDICTOR (cached)
# ════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI model …")
def load_predictor():
    from models.predictor import FakeJobPredictor
    return FakeJobPredictor()


@st.cache_resource(show_spinner="Loading explainability engines …")
def load_explainer():
    from explainability.explainer import JobExplainer
    return JobExplainer(load_predictor())


# ════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════

def gauge_chart(score: float) -> go.Figure:
    """Scam probability gauge."""
    color = "#e74c3c" if score >= 0.5 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score * 100,
        title = {"text": "Scam Probability (%)", "font": {"size": 14, "color": "#aaa"}},
        number= {"suffix": "%", "font": {"size": 28, "color": color}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "#555", "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1d2e",
            "bordercolor": "#2d3056",
            "steps": [
                {"range": [0,  40], "color": "rgba(46,204,113,0.15)"},
                {"range": [40, 65], "color": "rgba(243,156,18,0.15)"},
                {"range": [65,100], "color": "rgba(231,76,60,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    return fig


def score_bar(label: str, value: float, max_val: float = 1.0) -> go.Figure:
    """Horizontal bar for score breakdown."""
    color = "#e74c3c" if value >= 0.5 else "#2ecc71"
    fig = go.Figure(go.Bar(
        x=[value], y=[label],
        orientation="h",
        marker_color=color,
        text=[f"{value:.3f}"],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], showgrid=False, color="#555"),
        yaxis=dict(showgrid=False, color="#ccc"),
        height=60, margin=dict(l=0, r=40, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#ccc", size=11),
    )
    return fig


def highlight_text(text: str, keywords: list[str]) -> str:
    """Wrap suspicious keywords in a red highlight span."""
    from config import SUSPICIOUS_KEYWORDS
    all_kw = list(set(keywords + SUSPICIOUS_KEYWORDS))
    result = text
    for kw in sorted(all_kw, key=len, reverse=True):
        if kw.lower() in result.lower():
            import re
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            result  = pattern.sub(
                f'<mark style="background:#7a1a1a;color:#ffb3b3;border-radius:3px;padding:1px 3px">{kw}</mark>',
                result,
            )
    return result


def render_verdict_card(result: dict):
    """Render the main verdict card."""
    is_fake = result["prediction"] == "Fake"
    cls     = "verdict-fake" if is_fake else "verdict-real"
    color   = "fake-color" if is_fake else "real-color"
    icon    = "🚨" if is_fake else "✅"

    st.markdown(f"""
    <div class="{cls}">
      <p class="verdict-title {color}">{icon} {result["prediction"].upper()}</p>
      <p style="color:#aaa;margin:4px 0 0 0;font-size:0.9rem">
        This job posting appears to be <strong class="{color}">{result["prediction"].lower()}</strong>
        with a scam probability of <strong class="{color}">{result["scam_probability"]:.1%}</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_keywords(keywords: list[str]):
    """Render keyword pills."""
    if not keywords:
        st.info("No suspicious keywords detected.")
        return
    pills = "".join(f'<span class="kw-pill">⚠ {kw}</span>' for kw in keywords)
    st.markdown(f'<div style="margin:8px 0">{pills}</div>', unsafe_allow_html=True)


def render_signals(signals: list[dict]):
    """Render heuristic signal badges."""
    if not signals:
        st.success("No heuristic rule violations.")
        return
    for sig in signals:
        rule    = sig.get("rule", "").replace("_", " ").title()
        details = sig.get("details", "")
        weight  = sig.get("weight", 0)
        st.markdown(
            f'<span class="signal-badge">🔴 {rule} — {details} (w={weight:.2f})</span>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔍 FakeJob Detector")
    st.markdown("*DistilBERT + Heuristics*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔎 Single Analysis", "📁 Bulk Upload", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Model Settings**")
    show_lime = st.toggle("LIME Explanation", value=False, help="Slower but provides word-level LIME scores")
    show_shap = st.toggle("SHAP Explanation", value=False, help="Slower but provides SHAP token importances")
    show_tokens = st.toggle("Token Attributions", value=True)
    confidence_thresh = st.slider("Fake Threshold", 0.3, 0.8, 0.5, 0.05,
                                  help="Scores above this are labelled Fake")

    st.divider()
    st.markdown("**Hybrid Score Formula**")
    st.code("0.7 × BERT + 0.3 × Heuristic", language="python")
    st.markdown("---")
    st.caption("v1.0 · Built with DistilBERT + FastAPI + Streamlit")


# ════════════════════════════════════════════════════
# PAGE: SINGLE ANALYSIS
# ════════════════════════════════════════════════════

if "🔎 Single Analysis" in page:
    st.markdown("## 🔎 Analyse a Job Posting")
    st.markdown("Paste any job description below and hit **Analyse**.")

    # Sample buttons
    col_s1, col_s2, col_s3 = st.columns(3)
    if col_s1.button("Load Fake Example"):
        st.session_state["sample_text"] = (
            "URGENT HIRING!!! Work from home and earn $5,000/week! "
            "No interview needed, no experience required. "
            "Registration fee of $99 required to process your application. "
            "WhatsApp us NOW: +1-555-0199. Limited spots — act now! "
            "Guaranteed income, passive earnings, financial freedom!"
        )
    if col_s2.button("Load Real Example"):
        st.session_state["sample_text"] = (
            "TechCorp is hiring a Senior Software Engineer. "
            "Salary: $120,000–$150,000/year plus stock options. "
            "5+ years of Python/Java experience required. "
            "We offer full health benefits, 401k matching, and flexible PTO. "
            "Apply via our website with your CV and two references."
        )
    if col_s3.button("Clear"):
        st.session_state["sample_text"] = ""

    default_text = st.session_state.get("sample_text", "")
    job_text = st.text_area(
        "Job Description",
        value=default_text,
        height=180,
        placeholder="Paste job posting text here …",
        label_visibility="collapsed",
    )

    analyse_btn = st.button("🔍 Analyse", use_container_width=False)

    if analyse_btn and job_text.strip():
        predictor = load_predictor()

        with st.spinner("Analysing job posting …"):
            result = predictor.predict(job_text)

        # Apply custom threshold
        if result["scam_probability"] >= confidence_thresh:
            result["prediction"] = "Fake"
        else:
            result["prediction"] = "Real"

        st.markdown("---")

        # ── Verdict + gauge ─────────────────────────
        left, right = st.columns([2, 1])
        with left:
            render_verdict_card(result)
            st.markdown(f"*{result['explanation']}*")

        with right:
            st.plotly_chart(gauge_chart(result["scam_probability"]),
                            use_container_width=True, config={"displayModeBar": False})

        # ── Score breakdown ──────────────────────────
        st.markdown("#### Score Breakdown")
        s1, s2, s3 = st.columns(3)
        s1.metric("Hybrid Scam Score", f"{result['scam_probability']:.4f}")
        s2.metric("DistilBERT Score",  f"{result['bert_score']:.4f}")
        s3.metric("Heuristic Score",   f"{result['heuristic_score']:.4f}")

        # ── Score bars ──────────────────────────────
        for lbl, val in [
            ("Hybrid Score",    result["scam_probability"]),
            ("DistilBERT",      result["bert_score"]),
            ("Heuristic Rules", result["heuristic_score"]),
        ]:
            st.plotly_chart(score_bar(lbl, val),
                            use_container_width=True, config={"displayModeBar": False})

        # ── Highlighted text ─────────────────────────
        st.markdown("#### Highlighted Text")
        highlighted = highlight_text(job_text, result["keywords"])
        st.markdown(
            f'<div style="background:#1a1d2e;border:1px solid #2d3056;border-radius:8px;'
            f'padding:16px;font-size:0.9rem;line-height:1.7;color:#ddd">{highlighted}</div>',
            unsafe_allow_html=True,
        )

        # ── Keywords ─────────────────────────────────
        st.markdown("#### 🚩 Suspicious Keywords")
        render_keywords(result["keywords"])

        # ── Heuristic signals ─────────────────────────
        st.markdown("#### 🛡️ Heuristic Rule Violations")
        render_signals(result["signals"])

        # ── Token attributions ────────────────────────
        if show_tokens and result.get("token_attributions"):
            st.markdown("#### 🧠 Token Attributions (Gradient-based)")
            top_tokens = result["token_attributions"][:15]
            tokens_df = pd.DataFrame(top_tokens)
            fig = px.bar(
                tokens_df, x="score", y="token", orientation="h",
                color="score", color_continuous_scale=["#2ecc71", "#e67e22", "#e74c3c"],
                labels={"score": "Importance", "token": "Token"},
                height=400,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,29,46,0.6)",
                font_color="#ccc",
                yaxis_autorange="reversed",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── LIME ──────────────────────────────────────
        if show_lime:
            with st.spinner("Running LIME (this may take 20–40 seconds) …"):
                try:
                    explainer   = load_explainer()
                    lime_result = explainer.lime.explain(job_text)
                    if lime_result.get("features"):
                        st.markdown("#### 🟡 LIME Explanation")
                        ldf = pd.DataFrame(lime_result["features"])
                        fig_lime = px.bar(
                            ldf, x="weight", y="word", orientation="h",
                            color="weight",
                            color_continuous_scale=["#2ecc71", "#e67e22", "#e74c3c"],
                            height=400,
                        )
                        fig_lime.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(26,29,46,0.6)",
                            font_color="#ccc",
                            yaxis_autorange="reversed",
                        )
                        st.plotly_chart(fig_lime, use_container_width=True)
                except Exception as e:
                    st.warning(f"LIME explanation failed: {e}")

        # ── SHAP ──────────────────────────────────────
        if show_shap:
            with st.spinner("Running SHAP (this may take 30–60 seconds) …"):
                try:
                    explainer   = load_explainer()
                    shap_result = explainer.shap.explain(job_text)
                    if shap_result.get("values"):
                        st.markdown("#### 🔵 SHAP Explanation")
                        sdf = pd.DataFrame(shap_result["values"])
                        fig_shap = px.bar(
                            sdf, x="shap_value", y="token", orientation="h",
                            color="shap_value",
                            color_continuous_scale=["#2ecc71", "#e67e22", "#e74c3c"],
                            height=400,
                        )
                        fig_shap.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(26,29,46,0.6)",
                            font_color="#ccc",
                            yaxis_autorange="reversed",
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP explanation failed: {e}")

        # ── Download PDF report ───────────────────────
        st.markdown("---")
        st.markdown("#### 📥 Download Report")
        try:
            from backend.report_generator import generate_pdf_report
            pdf_bytes = generate_pdf_report(result, job_text)
            if pdf_bytes:
                st.download_button(
                    label     = "⬇️ Download PDF Report",
                    data      = pdf_bytes,
                    file_name = "fake_job_report.pdf",
                    mime      = "application/pdf",
                )
        except Exception as e:
            st.warning(f"PDF generation not available: {e}")

        # Download JSON
        st.download_button(
            label     = "⬇️ Download JSON",
            data      = json.dumps({k: v for k, v in result.items() if k != "token_attributions"}, indent=2),
            file_name = "fake_job_result.json",
            mime      = "application/json",
        )

    elif analyse_btn:
        st.warning("Please enter a job description.")


# ════════════════════════════════════════════════════
# PAGE: BULK UPLOAD
# ════════════════════════════════════════════════════

elif "📁 Bulk Upload" in page:
    st.markdown("## 📁 Bulk CSV Analysis")
    st.markdown(
        "Upload a CSV file with a **`text`** or **`description`** column "
        "containing job postings."
    )

    # Sample CSV download
    sample_data = pd.DataFrame({
        "title":       ["Software Engineer", "Work From Home Job", "Marketing Manager"],
        "text": [
            "TechCorp is hiring a Senior Engineer. Salary $120k/year. Apply with CV.",
            "EARN $5000/WEEK!!! No interview needed. Registration fee $99. WhatsApp now!",
            "Join our marketing team. 3+ years experience. Competitive benefits offered.",
        ],
    })
    st.download_button(
        "⬇️ Download Sample CSV",
        data      = sample_data.to_csv(index=False),
        file_name = "sample_jobs.csv",
        mime      = "text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_raw)} rows loaded.** Preview:")
        st.dataframe(df_raw.head(5), use_container_width=True)

        text_col = None
        for cand in ("text", "description", "job_description", "content"):
            if cand in df_raw.columns:
                text_col = cand
                break

        if text_col is None:
            st.error(f"No text column found. Columns: {list(df_raw.columns)}")
        else:
            if st.button("🚀 Run Bulk Analysis", use_container_width=False):
                predictor = load_predictor()
                texts     = df_raw[text_col].fillna("").tolist()

                progress = st.progress(0, text="Analysing …")
                results  = []
                for i, text in enumerate(texts):
                    results.append(predictor.predict(text))
                    progress.progress((i + 1) / len(texts),
                                      text=f"Processing {i+1}/{len(texts)} …")
                progress.empty()

                df_result = df_raw.copy()
                df_result["prediction"]       = [r["prediction"]       for r in results]
                df_result["scam_probability"] = [r["scam_probability"]  for r in results]
                df_result["bert_score"]       = [r["bert_score"]        for r in results]
                df_result["heuristic_score"]  = [r["heuristic_score"]   for r in results]
                df_result["keywords"]         = [", ".join(r["keywords"]) for r in results]
                df_result["explanation"]      = [r["explanation"]       for r in results]

                # ── Summary metrics ──────────────────
                st.markdown("---")
                st.markdown("### 📊 Bulk Results Summary")
                total    = len(results)
                fake_cnt = sum(1 for r in results if r["prediction"] == "Fake")
                real_cnt = total - fake_cnt
                avg_scr  = np.mean([r["scam_probability"] for r in results])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Jobs",        total)
                m2.metric("🚨 Fake",           fake_cnt,  delta=f"{fake_cnt/total:.0%}")
                m3.metric("✅ Real",            real_cnt,  delta=f"{real_cnt/total:.0%}")
                m4.metric("Avg Scam Score",    f"{avg_scr:.3f}")

                # ── Pie chart ────────────────────────
                fig_pie = go.Figure(go.Pie(
                    labels=["Real", "Fake"],
                    values=[real_cnt, fake_cnt],
                    marker_colors=["#2ecc71", "#e74c3c"],
                    hole=0.55,
                ))
                fig_pie.update_layout(
                    title_text="Real vs Fake Distribution",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # ── Score histogram ───────────────────
                scores = [r["scam_probability"] for r in results]
                fig_hist = px.histogram(
                    x=scores, nbins=20,
                    color_discrete_sequence=["#7c83fd"],
                    labels={"x": "Scam Probability"},
                    title="Score Distribution",
                )
                fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c",
                                   annotation_text="Threshold", annotation_position="top right")
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(26,29,46,0.6)",
                    font_color="#ccc", height=280,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # ── Results table ─────────────────────
                st.markdown("### Full Results")
                color_map = {"Fake": "background-color: rgba(231,76,60,0.2)",
                             "Real": "background-color: rgba(46,204,113,0.1)"}
                display_cols = [text_col, "prediction", "scam_probability",
                                "bert_score", "heuristic_score", "keywords"]
                st.dataframe(
                    df_result[[c for c in display_cols if c in df_result.columns]],
                    use_container_width=True,
                    height=400,
                )

                # ── Downloads ────────────────────────
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                dl1, dl2 = st.columns(2)
                dl1.download_button(
                    "⬇️ Download Full CSV",
                    data      = df_result.to_csv(index=False),
                    file_name = "bulk_fake_job_results.csv",
                    mime      = "text/csv",
                )
                # High-risk only
                high_risk = df_result[df_result["scam_probability"] >= 0.5]
                dl2.download_button(
                    f"⬇️ High-Risk Only ({len(high_risk)} rows)",
                    data      = high_risk.to_csv(index=False),
                    file_name = "high_risk_jobs.csv",
                    mime      = "text/csv",
                )


# ════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ════════════════════════════════════════════════════

elif "📊 Analytics" in page:
    st.markdown("## 📊 Analytics Dashboard")

    # Try to load training metrics
    try:
        from config import MODEL_DIR
        metrics_path = MODEL_DIR / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            st.markdown("### Model Performance on Test Set")
            cols = st.columns(5)
            for col, (key, label) in zip(cols, [
                ("accuracy",  "Accuracy"),
                ("precision", "Precision"),
                ("recall",    "Recall"),
                ("f1",        "F1-Score"),
                ("roc_auc",   "ROC-AUC"),
            ]):
                val = metrics.get(key, 0)
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-number">{val:.3f}</div>
                  <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

            # Confusion matrix
            cm = metrics.get("confusion_matrix")
            if cm:
                st.markdown("### Confusion Matrix")
                z = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]
                fig_cm = go.Figure(go.Heatmap(
                    z=z,
                    x=["Predicted: Fake", "Predicted: Real"],
                    y=["Actual: Fake",    "Actual: Real"],
                    colorscale=[[0, "#1a1d2e"], [1, "#7c83fd"]],
                    text=z, texttemplate="%{text}",
                    showscale=False,
                ))
                fig_cm.update_layout(
                    height=320,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        else:
            st.info("No trained model found. Run `python models/train_model.py` to train.")

    except Exception as e:
        st.warning(f"Could not load metrics: {e}")

    # Keyword frequency from config
    st.markdown("### Common Scam Keywords")
    from config import SUSPICIOUS_KEYWORDS
    # Simulate counts for demo
    import random
    random.seed(42)
    kw_data = pd.DataFrame({
        "keyword": SUSPICIOUS_KEYWORDS[:15],
        "frequency": [random.randint(5, 120) for _ in range(15)]
    }).sort_values("frequency", ascending=True)

    fig_kw = px.bar(
        kw_data, x="frequency", y="keyword", orientation="h",
        color="frequency", color_continuous_scale=["#5c63d8", "#e74c3c"],
        title="Top Suspicious Keywords (demo frequencies)",
    )
    fig_kw.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font_color="#ccc", height=500,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_kw, use_container_width=True)


# ════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════

elif "ℹ️ About" in page:
    st.markdown("## ℹ️ About the System")

    st.markdown("""
    ### Fake Job Detection System

    This system uses a **hybrid AI + rule-based approach** to detect fraudulent job postings.

    #### Architecture
    | Component | Technology |
    |-----------|-----------|
    | NLP Model | DistilBERT (fine-tuned) |
    | Framework | PyTorch + HuggingFace |
    | Heuristics | Custom rule engine |
    | Explainability | SHAP + LIME |
    | API | FastAPI |
    | Frontend | Streamlit |
    | Reports | PDF (fpdf2) + CSV |

    #### Hybrid Scoring
    ```
    final_score = 0.7 × DistilBERT_score + 0.3 × heuristic_score
    ```

    #### Heuristic Rules
    - **Suspicious keywords** — registration fee, urgent hiring, no interview, etc.
    - **Unrealistic salary** — suspiciously high (>$50k/month) or low (<$50/month)
    - **Missing company info** — no company name, "confidential employer"
    - **Upfront payment** — wire transfer, Western Union, fee required
    - **Excitement spam** — excessive caps, exclamation marks

    #### Dataset
    - [EMSCAD / Fake Job Postings](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction) (Kaggle)
    - ~17,880 job postings, ~4.8% fraudulent

    #### Training
    - **Model**: `distilbert-base-uncased`
    - **Epochs**: 3
    - **Batch size**: 16
    - **Optimizer**: AdamW (lr=2e-5)
    - **GPU**: auto-detected

    #### API Endpoints
    - `GET  /health`        — system status
    - `POST /predict`       — single prediction
    - `POST /bulk-predict`  — CSV batch
    - `GET  /metrics`       — model metrics
    - `GET  /report/{id}`   — download report
    """)

    st.markdown("---")
    st.markdown("**To run the API server:**")
    st.code("uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload", language="bash")
    st.markdown("**To train the model:**")
    st.code("python models/train_model.py", language="bash")
    st.markdown("**To run this UI:**")
    st.code("streamlit run frontend/app.py", language="bash")