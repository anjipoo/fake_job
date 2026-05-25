"""
explainability/explainer.py
============================
SHAP and LIME explanations for the Fake Job Detection system.
Both produce token/word-level importance scores with visualisations.
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import REPORT_DIR


class LIMEExplainer:
    def __init__(self,predictor):
        self.predictor=predictor

    def _predict_fn(self,texts:list[str])->np.ndarray:
        results=[]

        for text in texts:
            res=self.predictor.predict(text)

            p_fake=res["scam_probability"]

            results.append([1-p_fake,p_fake])

        return np.array(results)

    def explain(self,text:str,num_features:int=10)->dict:
        try:
            from lime.lime_text import LimeTextExplainer

        except ImportError:
            logger.error("lime not installed. Run: pip install lime")

            return {
                "error":"lime not installed",
                "features":[]
            }

        explainer=LimeTextExplainer(
            class_names=["Real","Fake"]
        )

        explanation=explainer.explain_instance(
            text,
            self._predict_fn,
            num_features=num_features,
            num_samples=200,
            labels=[1],
        )

        features=explanation.as_list(label=1)

        plot_path=self._plot(features,text[:60])

        return {
            "method":"LIME",
            "label":"Fake",
            "features":[
                {
                    "word":w,
                    "weight":round(float(v),4)
                }
                for w,v in features
            ],
            "plot_path":str(plot_path) if plot_path else None,
        }

    def _plot(self,features:list[tuple],title_snippet:str)->Optional[Path]:
        if not features:
            return None

        words=[f[0] for f in features]

        weights=[f[1] for f in features]

        colors=[
            "#e74c3c" if w>0 else "#2ecc71"
            for w in weights
        ]

        fig,ax=plt.subplots(
            figsize=(8,max(4,len(features)*0.4+1))
        )

        bars=ax.barh(
            words,
            weights,
            color=colors,
            edgecolor="white",
            height=0.65
        )

        ax.axvline(
            0,
            color="grey",
            linewidth=0.8,
            linestyle="--"
        )

        ax.set_xlabel(
            "LIME Weight  (positive → towards Fake)",
            fontsize=9
        )

        ax.set_title(
            f"LIME Explanation\n\"{title_snippet}…\"",
            fontsize=10,
            pad=8
        )

        ax.invert_yaxis()

        red_patch=mpatches.Patch(
            color="#e74c3c",
            label="Fake signal"
        )

        green_patch=mpatches.Patch(
            color="#2ecc71",
            label="Real signal"
        )

        ax.legend(
            handles=[red_patch,green_patch],
            fontsize=8
        )

        plt.tight_layout()

        out=REPORT_DIR/"lime_explanation.png"

        plt.savefig(
            out,
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        return out


class SHAPExplainer:
    def __init__(self,predictor):
        self.predictor=predictor

    def _predict_fn(self,texts)->np.ndarray:
        results=[]

        for text in texts:
            res=self.predictor.predict(text)

            p_fake=res["scam_probability"]

            results.append([1-p_fake,p_fake])

        return np.array(results)

    def explain(self,text:str,max_evals:int=100)->dict:
        try:
            import shap

        except ImportError:
            logger.error("shap not installed. Run: pip install shap")

            return {
                "error":"shap not installed",
                "values":[]
            }

        masker=shap.maskers.Text(
            tokenizer=r"\W+"
        )

        explainer=shap.Explainer(
            self._predict_fn,
            masker,
            output_names=["Real","Fake"]
        )

        try:
            shap_values=explainer(
                [text],
                max_evals=max_evals,
                silent=True
            )

        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

            return {
                "error":str(e),
                "values":[]
            }

        tokens=shap_values.data[0]

        values=shap_values.values[0,:,1]

        paired=sorted(
            [
                {
                    "token":t,
                    "shap_value":round(float(v),4)
                }
                for t,v in zip(tokens,values)
            ],
            key=lambda x:abs(x["shap_value"]),
            reverse=True,
        )[:20]

        plot_path=self._plot(paired,text[:60])

        return {
            "method":"SHAP",
            "label":"Fake",
            "values":paired,
            "plot_path":str(plot_path) if plot_path else None,
        }

    def _plot(self,pairs:list[dict],title_snippet:str)->Optional[Path]:
        if not pairs:
            return None

        labels=[p["token"] for p in pairs[:15]]

        vals=[p["shap_value"] for p in pairs[:15]]

        colors=[
            "#c0392b" if v>0 else "#27ae60"
            for v in vals
        ]

        fig,ax=plt.subplots(
            figsize=(8,max(4,len(labels)*0.4+1))
        )

        ax.barh(
            labels,
            vals,
            color=colors,
            edgecolor="white",
            height=0.65
        )

        ax.axvline(
            0,
            color="grey",
            linewidth=0.8,
            linestyle="--"
        )

        ax.set_xlabel(
            "SHAP Value  (positive → towards Fake)",
            fontsize=9
        )

        ax.set_title(
            f"SHAP Explanation\n\"{title_snippet}…\"",
            fontsize=10,
            pad=8
        )

        ax.invert_yaxis()

        red_patch=mpatches.Patch(
            color="#c0392b",
            label="Fake signal"
        )

        green_patch=mpatches.Patch(
            color="#27ae60",
            label="Real signal"
        )

        ax.legend(
            handles=[red_patch,green_patch],
            fontsize=8
        )

        plt.tight_layout()

        out=REPORT_DIR/"shap_explanation.png"

        plt.savefig(
            out,
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        return out


class JobExplainer:
    def __init__(self,predictor):
        self.lime=LIMEExplainer(predictor)

        self.shap=SHAPExplainer(predictor)

    def explain(self,text:str)->dict:
        logger.info("Running LIME …")

        lime_result=self.lime.explain(text)

        logger.info("Running SHAP …")

        shap_result=self.shap.explain(
            text,
            max_evals=80
        )

        return {
            "lime":lime_result,
            "shap":shap_result,
        }


if __name__=="__main__":
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

    from models.predictor import FakeJobPredictor

    predictor=FakeJobPredictor()

    explainer=JobExplainer(predictor)

    text=(
        "URGENT HIRING!!! Earn $5000/week from home! "
        "No interview needed. Registration fee $99. Whatsapp now!"
    )

    result=explainer.explain(text)

    print("\nLIME top features:")

    for f in result["lime"].get("features",[])[:5]:
        print(f"  {f['word']:20s}  {f['weight']:+.4f}")

    print("\nSHAP top tokens:")

    for v in result["shap"].get("values",[])[:5]:
        print(f"  {v['token']:20s}  {v['shap_value']:+.4f}")