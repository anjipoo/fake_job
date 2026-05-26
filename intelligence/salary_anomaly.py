import re
import sys
import math
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import MODEL_DIR,SEED

ROLE_BENCHMARKS={
    "executive":(8_000,50_000,"Executive"),
    "director":(7_000,40_000,"Director"),
    "manager":(5_000,20_000,"Manager"),
    "engineer":(5_000,18_000,"Engineer"),
    "developer":(4_500,17_000,"Developer"),
    "data scientist":(5_000,18_000,"Data Scientist"),
    "analyst":(3_500,12_000,"Analyst"),
    "designer":(3_000,12_000,"Designer"),
    "sales":(2_500,15_000,"Sales"),
    "support":(2_000,7_000,"Support"),
    "intern":(500,3_000,"Intern"),
    "entry":(1_500,6_000,"Entry-level"),
    "fresher":(300,3_000,"Fresher"),
    "default":(1_500,20_000,"General"),
}

CURRENCY_TO_USD={
    "$":1.0,
    "usd":1.0,
    "₹":0.012,
    "inr":0.012,
    "£":1.27,
    "gbp":1.27,
    "€":1.08,
    "eur":1.08,
    "aed":0.27,
    "lpa":833.33/12*0.012,
}

PERIOD_TO_MONTHLY={
    "hour":160,
    "hr":160,
    "hourly":160,
    "day":22,
    "daily":22,
    "week":4,
    "weekly":4,
    "month":1,
    "monthly":1,
    "year":1/12,
    "annual":1/12,
    "annually":1/12,
    "pa":1/12,
    "lpa":1/12,
}

SALARY_PATTERNS=[
    re.compile(
        r"(?P<currency>[₹$£€]|INR|USD|GBP|EUR|AED|LPA)\s*"
        r"(?P<amount>[\d,]+(?:\.\d+)?)\s*"
        r"(?:k|K|lakh|l|L)?\s*"
        r"(?:/|per|a)?\s*"
        r"(?P<period>hour|hr|day|week|month|year|annual|annually|lpa|pa)?",
        re.IGNORECASE
    ),

    re.compile(
        r"(?P<amount>[\d,]+(?:\.\d+)?)\s*"
        r"(?:k|K)?\s*"
        r"(?:/|per|a)?\s*"
        r"(?P<period>hour|hr|day|week|month|year|annual|annually|lpa|pa)",
        re.IGNORECASE
    ),
]

INSTANT_EARN_PATTERNS=[
    r"earn\s+\$[\d,]+\s*(instantly|immediately|today|daily|guaranteed)",
    r"make\s+\$[\d,]+\s*\w*\s*from\s+home",
    r"guaranteed\s+income\s+of\s+\$[\d,]+",
    r"₹\s*\d+\s*lpa\s+no\s+experience",
    r"get\s+paid\s+\$[\d,]+\s*(per|a)?\s*day",
    r"passive\s+income\s+\$[\d,]+",
    r"\$[\d,]+\s*\/?\s*week\s+(no|without)\s+experience",
]

class SalaryParser:

    def parse(self,text:str)->list[dict]:
        results=[]
        seen=set()

        for pattern in SALARY_PATTERNS:
            for m in pattern.finditer(text):
                raw_amount_str=m.group("amount").replace(",","")

                try:
                    raw_amount=float(raw_amount_str)
                except ValueError:
                    continue

                full_match=m.group(0)

                if re.search(r"\d\s*[kK]",full_match):
                    raw_amount*=1000

                if re.search(r"\d\s*(lakh|l\b)",full_match,re.IGNORECASE):
                    raw_amount*=100_000

                period=(m.groupdict().get("period") or "month").lower()
                currency=(m.groupdict().get("currency") or "$").lower()

                cur_key=currency.replace(" ","")
                usd_rate=CURRENCY_TO_USD.get(cur_key,1.0)

                per_key=period.replace(" ","")
                per_mult=PERIOD_TO_MONTHLY.get(per_key,1.0)

                monthly_usd=raw_amount*usd_rate*per_mult

                key=round(monthly_usd)

                if key in seen or key==0:
                    continue

                seen.add(key)

                results.append({
                    "raw":m.group(0).strip(),
                    "raw_amount":raw_amount,
                    "currency":currency,
                    "period":period,
                    "amount_usd_monthly":round(monthly_usd,2),
                })

        return results

class SalaryAnomalyModels:

    IF_PATH=MODEL_DIR/"salary_isolation_forest.pkl"
    LOF_PATH=MODEL_DIR/"salary_lof.pkl"

    def __init__(self):
        self.iso_forest=None
        self.lof=None
        self._load_or_train()

    @staticmethod
    def _make_training_salaries()->np.ndarray:
        rng=np.random.default_rng(SEED)
        samples=[]

        for mu,sigma,n in [
            (3000,800,200),
            (6000,1500,300),
            (10000,2500,200),
            (18000,4000,100),
            (800,200,100),
        ]:
            samples.extend(rng.normal(mu,sigma,n).tolist())

        samples.extend([50_000,80_000,200_000,500_000])
        samples.extend([5,10,20])

        return np.clip(np.array(samples),1,None).reshape(-1,1)

    def _load_or_train(self):
        if self.IF_PATH.exists() and self.LOF_PATH.exists():
            try:
                with open(self.IF_PATH,"rb") as f:
                    self.iso_forest=pickle.load(f)

                with open(self.LOF_PATH,"rb") as f:
                    self.lof=pickle.load(f)

                logger.info("Salary anomaly models loaded.")
                return

            except Exception as e:
                logger.warning(f"Could not load salary models: {e}")

        self._train()

    def _train(self):
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor

        except ImportError:
            logger.error("scikit-learn required for salary anomaly detection.")
            return

        X=self._make_training_salaries()

        self.iso_forest=IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=SEED,
            n_jobs=-1
        )

        self.iso_forest.fit(X)

        self.lof=LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True
        )

        self.lof.fit(X)

        with open(self.IF_PATH,"wb") as f:
            pickle.dump(self.iso_forest,f)

        with open(self.LOF_PATH,"wb") as f:
            pickle.dump(self.lof,f)

        logger.info("Salary anomaly models trained and saved.")

    def score(self,monthly_usd:float)->dict:
        X=np.array([[monthly_usd]])

        if_score=0.5
        lof_score=0.5

        if self.iso_forest is not None:
            raw=self.iso_forest.decision_function(X)[0]
            if_score=float(np.clip(1-(raw+0.5),0,1))

        if self.lof is not None:
            raw=self.lof.decision_function(X)[0]
            lof_score=float(np.clip(1-(raw+0.5),0,1))

        return {
            "isolation_forest_anomaly":round(if_score,4),
            "lof_anomaly":round(lof_score,4),
        }

class SalaryAnomalyDetector:

    def __init__(self):
        self.parser=SalaryParser()
        self.models=SalaryAnomalyModels()

    def analyse(self,text:str,role_hint:str="")->dict:

        red_flags=self._scan_red_flags(text)

        salaries=self.parser.parse(text)

        if not salaries and not red_flags:
            return self._no_salary(red_flags)

        benchmark=self._get_benchmark(role_hint or text)

        anomalies=[]
        max_score=0.0

        for sal in salaries:
            monthly=sal["amount_usd_monthly"]

            result=self._score_salary(monthly,benchmark)

            result.update(sal)

            anomalies.append(result)

            max_score=max(max_score,result["anomaly_score"])

        if red_flags:
            max_score=min(max_score+0.25*len(red_flags),1.0)

        risk_level=(
            "CRITICAL" if max_score>=0.80 else
            "HIGH" if max_score>=0.60 else
            "MEDIUM" if max_score>=0.35 else
            "LOW"
        )

        return {
            "salaries_found":salaries,
            "anomaly_score":round(max_score,4),
            "risk_level":risk_level,
            "anomalies":anomalies,
            "red_flags":red_flags,
            "benchmark":benchmark,
            "explanation":self._explain(
                max_score,
                anomalies,
                red_flags,
                benchmark
            ),
        }

    def _score_salary(self,monthly_usd:float,benchmark:dict)->dict:
        lo,hi=benchmark["min_monthly"],benchmark["max_monthly"]

        mid=(lo+hi)/2
        spread=(hi-lo)/2 or 1

        z_score=abs(monthly_usd-mid)/spread

        ml=self.models.score(monthly_usd)

        z_component=min(z_score/5.0,1.0)

        combined=(
            0.30*z_component+
            0.35*ml["isolation_forest_anomaly"]+
            0.35*ml["lof_anomaly"]
        )

        direction=(
            "WAY_TOO_HIGH" if monthly_usd>hi*3 else
            "TOO_HIGH" if monthly_usd>hi*1.5 else
            "TOO_LOW" if monthly_usd<lo*0.3 else
            "SLIGHTLY_LOW" if monthly_usd<lo else
            "NORMAL"
        )

        return {
            "anomaly_score":round(min(combined,1.0),4),
            "z_score":round(z_score,2),
            "direction":direction,
            **ml,
        }

    def _get_benchmark(self,text:str)->dict:
        text_lower=text.lower()

        for keyword,(lo,hi,label) in ROLE_BENCHMARKS.items():
            if keyword in text_lower:
                return {
                    "role":label,
                    "min_monthly":lo,
                    "max_monthly":hi,
                }

        lo,hi,label=ROLE_BENCHMARKS["default"]

        return {
            "role":label,
            "min_monthly":lo,
            "max_monthly":hi
        }

    @staticmethod
    def _scan_red_flags(text:str)->list[str]:
        hits=[]

        for pattern in INSTANT_EARN_PATTERNS:
            m=re.search(pattern,text,re.IGNORECASE)

            if m:
                hits.append(m.group(0).strip())

        return hits

    @staticmethod
    def _explain(score,anomalies,red_flags,benchmark):
        if score<0.35:
            return "Salary figures appear realistic for the role."

        parts=[f"Salary anomaly detected (score: {score:.0%})."]

        for a in anomalies[:2]:
            amt=a["amount_usd_monthly"]

            dir_=a["direction"].replace("_"," ").lower()

            parts.append(
                f"${amt:,.0f}/month is {dir_} "
                f"for a {benchmark['role']} role "
                f"(expected ${benchmark['min_monthly']:,}–${benchmark['max_monthly']:,})."
            )

        if red_flags:
            parts.append(
                f"Red-flag phrases found: \"{red_flags[0]}\"."
            )

        return " ".join(parts)

    @staticmethod
    def _no_salary(red_flags):
        score=0.3*len(red_flags) if red_flags else 0.0

        return {
            "salaries_found":[],
            "anomaly_score":round(min(score,1.0),4),
            "risk_level":"MEDIUM" if red_flags else "LOW",
            "anomalies":[],
            "red_flags":red_flags,
            "benchmark":ROLE_BENCHMARKS["default"],
            "explanation":(
                f"No parseable salary found. Red-flag phrases: {red_flags}"
                if red_flags else
                "No salary information found."
            ),
        }

if __name__=="__main__":
    detector=SalaryAnomalyDetector()

    tests=[
        (
            "normal",
            "Software Engineer, salary $7,000/month, 3 years experience required."
        ),
        (
            "high",
            "EARN $5000/week instantly! No experience needed. Work from home."
        ),
        (
            "INR",
            "Fresher opening: ₹15 LPA, no experience, immediate joining."
        ),
        (
            "exec",
            "CEO position. Compensation: $45,000/month + equity. 10 years required."
        ),
        (
            "too_low",
            "Customer support role. Pay: $1/hour. Full-time position."
        ),
    ]

    print("=== Salary Anomaly Detector Test ===\n")

    for label,text in tests:
        r=detector.analyse(text)

        print(f"[{label}]")
        print(
            f"  Anomaly Score : {r['anomaly_score']:.4f}  [{r['risk_level']}]"
        )
        print(
            f"  Salaries      : {[s['raw'] for s in r['salaries_found']]}"
        )
        print(f"  Red Flags     : {r['red_flags']}")
        print(f"  Explanation   : {r['explanation']}")
        print()