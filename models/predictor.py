import re
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import(
PRETRAINED_MODEL,SAVED_MODEL_PATH,MAX_TOKEN_LENGTH,
SUSPICIOUS_KEYWORDS,BERT_WEIGHT,HEURISTIC_WEIGHT,
HIGH_SALARY_THRESHOLD,LOW_SALARY_THRESHOLD,LABEL_NAMES
)

class HeuristicDetector:

    SALARY_RE=re.compile(
        r"\$\s*([\d,]+)\s*(?:per|/|a)?\s*(hour|hr|day|week|month|year|annually)?",
        re.IGNORECASE
    )

    MISSING_COMPANY_SIGNALS=[
        r"\bno company\b",r"\bconfidential\b",r"\bundisclosed\b",
        r"company:\s*n/a",r"employer:\s*unknown"
    ]

    UPFRONT_PATTERNS=[
        r"pay.*\$\d+",r"send.*\$\d+",r"fee.*required",
        r"western union",r"money order",r"wire transfer",
        r"bitcoin.*payment",r"zelle.*deposit"
    ]

    def analyse(self,text:str)->dict:
        text_lower=text.lower()
        signals=[]
        score=0.0

        kw_hits=[kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
        if kw_hits:
            keyword_score=min(len(kw_hits)*0.12,0.6)
            score+=keyword_score
            signals.append({
                "rule":"suspicious_keywords",
                "details":kw_hits,
                "weight":round(keyword_score,2)
            })

        salary_signal=self._check_salary(text)
        if salary_signal:
            score+=0.2
            signals.append(salary_signal)

        for pat in self.MISSING_COMPANY_SIGNALS:
            if re.search(pat,text_lower):
                score+=0.1
                signals.append({"rule":"missing_company_info","details":pat,"weight":0.1})
                break

        for pat in self.UPFRONT_PATTERNS:
            if re.search(pat,text_lower):
                score+=0.25
                signals.append({"rule":"upfront_payment","details":pat,"weight":0.25})
                break

        exclaim_count=text.count("!")
        caps_ratio=sum(1 for c in text if c.isupper())/max(len(text),1)

        if exclaim_count>3 or caps_ratio>0.3:
            score+=0.1
            signals.append({
                "rule":"excitement_spam",
                "details":f"exclamation_marks={exclaim_count},caps_ratio={caps_ratio:.2f}",
                "weight":0.1
            })

        return{
            "heuristic_score":round(min(score,1.0),4),
            "signals":signals,
            "keywords_found":kw_hits if kw_hits else []
        }

    def _check_salary(self,text:str)->Optional[dict]:
        for m in self.SALARY_RE.finditer(text):
            try:
                amount=float(m.group(1).replace(",",""))
                period=(m.group(2) or "").lower()

                monthly=amount
                if period in("hour","hr"):
                    monthly=amount*160
                elif period=="day":
                    monthly=amount*22
                elif period in("year","annually"):
                    monthly=amount/12

                if monthly>HIGH_SALARY_THRESHOLD or(0<monthly<50):
                    return{
                        "rule":"unrealistic_salary",
                        "details":f"${amount:,.0f}/{period or 'unspecified'}",
                        "weight":0.2
                    }
            except(ValueError,AttributeError):
                continue
        return None


class BERTPredictor:

    def __init__(self):
        self.model=None
        self.tokenizer=None
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        model_path=Path(SAVED_MODEL_PATH)

        if model_path.exists() and(model_path/"config.json").exists():
            try:
                from transformers import(
                    DistilBertForSequenceClassification,
                    DistilBertTokenizerFast
                )

                self.tokenizer=DistilBertTokenizerFast.from_pretrained(SAVED_MODEL_PATH)
                self.model=DistilBertForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)
                self.model.to(self.device)
                self.model.eval()

                logger.info(f"Fine-tuned model loaded from {SAVED_MODEL_PATH}")
                return

            except Exception as e:
                logger.warning(f"Could not load fine-tuned model:{e}")

        logger.warning(
            "Fine-tuned model not found. Using raw DistilBERT."
        )

        try:
            from transformers import(
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast
            )

            self.tokenizer=DistilBertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
            self.model=DistilBertForSequenceClassification.from_pretrained(
                PRETRAINED_MODEL,num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Could not load any BERT model:{e}")

    def predict_proba(self,text:str)->float:
        if self.model is None or self.tokenizer is None:
            return 0.5

        inputs=self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_TOKEN_LENGTH
        ).to(self.device)

        with torch.no_grad():
            logits=self.model(**inputs).logits

        probs=torch.softmax(logits,dim=-1).cpu().numpy()[0]
        return float(probs[1])

    def get_token_attributions(self,text:str)->list[dict]:
        if self.model is None or self.tokenizer is None:
            return []

        inputs=self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH
        ).to(self.device)

        self.model.eval()
        embeddings=self.model.distilbert.embeddings(inputs["input_ids"])
        embeddings.retain_grad()

        logits=self.model(
            inputs_embeds=embeddings,
            attention_mask=inputs["attention_mask"]
        ).logits

        fake_logit=logits[0,1]
        self.model.zero_grad()
        fake_logit.backward()

        grads=embeddings.grad[0].detach().cpu().numpy()
        scores=np.abs(grads).sum(axis=-1)
        tokens=self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        pairs=[
            {"token":t,"score":float(s)}
            for t,s in zip(tokens,scores)
            if t not in("[CLS]","[SEP]","[PAD]")
        ]

        pairs.sort(key=lambda x:x["score"],reverse=True)
        return pairs[:20]


class FakeJobPredictor:

    def __init__(self):
        self.bert=BERTPredictor()
        self.heuristic=HeuristicDetector()
        logger.info("FakeJobPredictor ready.")

    def predict(self,text:str)->dict:
        if not text or not text.strip():
            return self._empty_result()

        bert_score=self.bert.predict_proba(text)

        h=self.heuristic.analyse(text)
        heuristic_score=h["heuristic_score"]
        signals=h["signals"]
        keywords=h["keywords_found"]

        final_score=(BERT_WEIGHT*bert_score)+(HEURISTIC_WEIGHT*heuristic_score)
        final_score=round(min(max(final_score,0.0),1.0),4)

        prediction="Fake" if final_score>=0.5 else "Real"

        explanation=self._build_explanation(
            prediction,final_score,keywords,signals
        )

        token_attributions=self.bert.get_token_attributions(text[:512])

        return{
            "prediction":prediction,
            "scam_probability":final_score,
            "bert_score":round(bert_score,4),
            "heuristic_score":round(heuristic_score,4),
            "keywords":keywords,
            "signals":signals,
            "explanation":explanation,
            "token_attributions":token_attributions
        }

    def predict_bulk(self,texts:list[str])->list[dict]:
        return[self.predict(t) for t in texts]

    def _build_explanation(self,prediction:str,score:float,keywords:list[str],signals:list[dict])->str:
        if prediction=="Real":
            return f"This posting appears legitimate (scam probability:{score:.0%}). No significant red flags were detected."

        parts=[f"This posting was flagged as FAKE (scam probability:{score:.0%})."]

        if keywords:
            kw_str=", ".join(f'"{k}"' for k in keywords[:4])
            parts.append(f"Suspicious phrases detected:{kw_str}.")

        rule_names={s["rule"] for s in signals}

        if "upfront_payment" in rule_names:
            parts.append("Requests upfront payment.")
        if "unrealistic_salary" in rule_names:
            parts.append("Unrealistic salary detected.")
        if "missing_company_info" in rule_names:
            parts.append("Missing company information.")
        if "excitement_spam" in rule_names:
            parts.append("Spam-like formatting detected.")

        return" ".join(parts)

    @staticmethod
    def _empty_result()->dict:
        return{
            "prediction":"Unknown",
            "scam_probability":0.0,
            "bert_score":0.0,
            "heuristic_score":0.0,
            "keywords":[],
            "signals":[],
            "explanation":"No text provided.",
            "token_attributions":[]
        }


if __name__=="__main__":
    predictor=FakeJobPredictor()

    tests=[
        ("LEGITIMATE","We are hiring a Software Engineer at Acme Corp Salary $80k/year."),
        ("FAKE","URGENT!!! Earn $5000/week from home. Pay $99 registration fee now!!!")
    ]

    for label,text in tests:
        result=predictor.predict(text)
        print(f"\n[{label}]")
        print(result)