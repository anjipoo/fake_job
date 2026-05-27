import re
import sys
import math
import hashlib
import difflib
from pathlib import Path
from typing import Optional
from collections import Counter
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from config import FREE_EMAIL_PROVIDERS,SCAM_RECRUITER_PATTERNS


class RecruiterIdentityAnalyser:

    EMAIL_RE=re.compile(r"[\w.\-+]+@([\w.\-]+\.[a-zA-Z]{2,})",re.IGNORECASE)

    NAME_RE=re.compile(
        r"(?:contact|recruiter|hr|from|regards|sincerely)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        re.IGNORECASE
    )

    PHONE_RE=re.compile(r"(?:\+?[\d\s\-().]{7,15}\d)")

    GENERIC_NAME_PATTERNS=[
        r"^(mr|mrs|ms|miss|dr)\.?\s+[a-z]+$",
        r"hr\s*(manager|executive|team|dept)\s*\d*",
        r"recruitment\s*(team|cell|unit|agent)\s*\d*",
        r"(jobs?|hiring|careers?|staffing)\s*(team|dept|division)",
        r"[a-z]+\d{3,}",
        r"^(admin|support|info|noreply|no-reply)$",
    ]

    def analyse(self,text:str,recruiter_name:str="",
                recruiter_email:str="")->dict:

        signals=[]
        score=0.0

        emails=self.EMAIL_RE.findall(text.lower())

        if recruiter_email:
            emails.insert(0,recruiter_email.lower().split("@")[-1])

        is_free=any(e in FREE_EMAIL_PROVIDERS for e in emails)

        is_disp=any(
            k in " ".join(emails)
            for k in ["mailinator","yopmail","throwam","tempmail","guerrilla"]
        )

        if is_disp:
            score+=0.45
            signals.append({
                "signal":"disposable_email",
                "severity":"CRITICAL"
            })

        elif is_free:
            score+=0.25
            signals.append({
                "signal":"free_email_recruiter",
                "severity":"HIGH"
            })

        company_match=self._check_company_email_match(text,emails)

        if not company_match and emails:
            score+=0.15
            signals.append({
                "signal":"email_company_mismatch",
                "severity":"MEDIUM"
            })

        name=recruiter_name.lower() or self._extract_name(text)

        if name:
            for pat in self.GENERIC_NAME_PATTERNS:
                if re.match(pat,name,re.IGNORECASE):
                    score+=0.20
                    signals.append({
                        "signal":"generic_recruiter_name",
                        "detail":name,
                        "severity":"MEDIUM"
                    })
                    break

        for pat in SCAM_RECRUITER_PATTERNS:
            if re.search(pat,text,re.IGNORECASE):
                score+=0.25
                signals.append({
                    "signal":"scam_recruiter_pattern",
                    "detail":pat,
                    "severity":"HIGH"
                })
                break

        wa_only=bool(re.search(r"whatsapp",text,re.IGNORECASE))
        tg_only=bool(re.search(r"telegram|t\.me",text,re.IGNORECASE))

        no_official=not bool(
            re.search(r"@[\w.\-]+\.(com|org|io|co)",text)
        )

        if (wa_only or tg_only) and no_official:
            score+=0.30
            signals.append({
                "signal":"messaging_app_only_contact",
                "severity":"HIGH"
            })

        return {
            "emails_found":emails,
            "is_free_email":is_free,
            "is_disposable":is_disp,
            "recruiter_name":name,
            "identity_score":round(min(score,1.0),4),
            "signals":signals,
        }

    def _extract_name(self,text:str)->str:
        m=self.NAME_RE.search(text)
        return m.group(1).strip().lower() if m else ""

    @staticmethod
    def _check_company_email_match(text:str,email_domains:list[str])->bool:
        company_re=re.compile(
            r"(?:company|employer|organization|firm|corp)[:\s]+([A-Z][^\n.]{2,40})",
            re.IGNORECASE
        )

        m=company_re.search(text)

        if not m or not email_domains:
            return True

        company=re.sub(r"[^a-z]","",m.group(1).lower())

        for domain in email_domains:
            domain_clean=re.sub(r"[^a-z]","",domain.split(".")[0])

            if company[:6] in domain_clean or domain_clean[:6] in company:
                return True

        return False


class PostingPatternAnalyser:

    TEMPLATE_MARKERS=[
        r"dear\s+applicant[,.]",
        r"you\s+have\s+been\s+selected\s+for\s+an\s+interview",
        r"kindly\s+send\s+your\s+(cv|resume)\s+to",
        r"we\s+are\s+pleased\s+to\s+inform\s+you",
        r"no\s+experience\s+required.*guaranteed",
        r"limited\s+seats\s+available\s+apply\s+now",
        r"our\s+company\s+is\s+expanding\s+globally",
        r"send\s+your\s+details\s+to\s+get\s+started",
    ]

    def analyse_single(self,text:str)->dict:
        text_lower=text.lower()
        hits=[]

        for pat in self.TEMPLATE_MARKERS:
            if re.search(pat,text_lower):
                hits.append(pat)

        exclaim=text.count("!")
        caps_ratio=sum(c.isupper() for c in text)/max(len(text),1)

        template_score=min(len(hits)*0.18,0.72)

        if exclaim>3:
            template_score+=0.10

        if caps_ratio>0.25:
            template_score+=0.12

        return {
            "template_hits":hits,
            "exclaim_count":exclaim,
            "caps_ratio":round(caps_ratio,3),
            "template_score":round(min(template_score,1.0),4),
        }

    def similarity_score(self,text_a:str,text_b:str)->float:
        return round(
            difflib.SequenceMatcher(
                None,
                text_a.lower()[:1000],
                text_b.lower()[:1000]
            ).ratio(),
            4
        )

    def detect_duplicates(self,postings:list[str],
                          threshold:float=0.75)->list[dict]:

        pairs=[]

        for i in range(len(postings)):
            for j in range(i+1,len(postings)):
                sim=self.similarity_score(postings[i],postings[j])

                if sim>=threshold:
                    pairs.append({
                        "posting_a":i,
                        "posting_b":j,
                        "similarity":sim,
                    })

        return pairs


class PhoneClusterAnalyser:

    PHONE_RE=re.compile(r"(?:\+?[\d\s\-().]{7,15}\d)")

    def extract_phones(self,text:str)->list[str]:
        raw=self.PHONE_RE.findall(text)

        return [
            re.sub(r"[\s\-().+]","",p)
            for p in raw
            if len(re.sub(r"\D","",p))>=7
        ]

    def cluster_analysis(self,postings:list[str])->dict:

        phone_to_postings:dict[str,list[int]]={}

        for idx,text in enumerate(postings):
            for phone in self.extract_phones(text):
                phone_to_postings.setdefault(phone,[]).append(idx)

        shared={
            p:idxs
            for p,idxs in phone_to_postings.items()
            if len(idxs)>=2
        }

        return {
            "shared_phones":shared,
            "scam_clusters":len(shared),
            "cluster_score":min(len(shared)*0.20,1.0),
        }


class RecruiterRiskScorer:

    def __init__(self):
        self.identity=RecruiterIdentityAnalyser()
        self.pattern=PostingPatternAnalyser()
        self.phone=PhoneClusterAnalyser()

    def score(self,text:str,
              recruiter_name:str="",
              recruiter_email:str="",
              corpus:Optional[list[str]]=None)->dict:

        id_r=self.identity.analyse(text,recruiter_name,recruiter_email)
        pat_r=self.pattern.analyse_single(text)

        dup_r=[]

        phone_r={
            "scam_clusters":0,
            "cluster_score":0.0,
            "shared_phones":{}
        }

        if corpus:
            dup_r=self.pattern.detect_duplicates([text]+corpus)
            phone_r=self.phone.cluster_analysis([text]+corpus)

        risk_score=(
            0.50*id_r["identity_score"]+
            0.35*pat_r["template_score"]+
            0.15*phone_r["cluster_score"]
        )

        risk_score=round(min(risk_score,1.0),4)

        risk_level=(
            "CRITICAL" if risk_score>=0.80 else
            "HIGH" if risk_score>=0.60 else
            "MEDIUM" if risk_score>=0.35 else
            "LOW"
        )

        all_signals=[
            s["signal"] for s in id_r["signals"]
        ]+pat_r["template_hits"]

        return {
            "recruiter_risk_score":risk_score,
            "risk_level":risk_level,
            "identity":id_r,
            "pattern":pat_r,
            "duplicates":dup_r,
            "phone_clusters":phone_r,
            "signals":all_signals,
            "explanation":self._explain(
                risk_score,
                id_r,
                pat_r,
                dup_r
            ),
        }

    @staticmethod
    def _explain(score,id_r,pat_r,dup_r)->str:

        if score<0.35:
            return "Recruiter signals appear legitimate."

        parts=[f"Recruiter risk is ELEVATED (score: {score:.0%})."]

        if id_r.get("is_disposable"):
            parts.append("Disposable email address used.")

        elif id_r.get("is_free_email"):
            parts.append(
                "Free email provider (Gmail/Yahoo etc.) used instead of corporate email."
            )

        if pat_r["template_hits"]:
            parts.append(
                f"Copy-paste template patterns detected ({len(pat_r['template_hits'])} hits)."
            )

        if pat_r["exclaim_count"]>3:
            parts.append(
                f"Excessive exclamation marks ({pat_r['exclaim_count']}) — spam indicator."
            )

        if dup_r:
            parts.append(
                f"{len(dup_r)} near-duplicate posting(s) found in corpus."
            )

        return " ".join(parts)


if __name__=="__main__":

    scorer=RecruiterRiskScorer()

    legit=(
        "Acme Corp is hiring a Python Engineer. Contact Jane Smith at "
        "jane.smith@acmecorp.com or call +1-415-555-0100. "
        "Office: 500 Market St, San Francisco CA."
    )

    scam=(
        "URGENT HIRING!!! We are pleased to inform you that you have been selected. "
        "Dear Applicant, Kindly send your CV to hr_manager123@gmail.com. "
        "WhatsApp: +1-555-9999. Registration fee $50. Limited seats — Apply NOW!!!"
    )

    corpus=[
        "URGENT HIRING!!! Same amazing opportunity! WhatsApp +1-555-9999!!!",
        "Earn $5000 weekly, no experience. Contact hr_fake@yahoo.com",
    ]

    for label,text in [("LEGIT",legit),("SCAM",scam)]:

        r=scorer.score(
            text,
            corpus=corpus if label=="SCAM" else None
        )

        print(f"[{label}]")
        print(f"  Risk Score  : {r['recruiter_risk_score']:.4f}  [{r['risk_level']}]")
        print(f"  Signals     : {r['signals'][:4]}")
        print(f"  Explanation : {r['explanation']}")
        print()