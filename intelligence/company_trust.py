import re
import sys
import math
import hashlib
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from config import FREE_EMAIL_PROVIDERS,SUSPICIOUS_TLDS


class EmailSignalExtractor:

    EMAIL_RE=re.compile(r"[\w.\-+]+@([\w.\-]+\.[a-zA-Z]{2,})",re.IGNORECASE)

    def extract(self,text:str)->dict:
        emails=self.EMAIL_RE.findall(text.lower())
        domains=list(set(emails))

        if not domains:
            return {
                "emails_found":[],
                "free_provider":False,
                "disposable":False,
                "corporate_domain":False,
                "email_score":0.5,
                "signals":["no_email_found"],
            }

        disposable_kw={"mailinator","guerrilla","yopmail","throwam",
                       "tempmail","sharklasers","trashmail","fakeinbox"}

        is_free=any(d in FREE_EMAIL_PROVIDERS for d in domains)
        is_disposable=any(any(k in d for k in disposable_kw) for d in domains)
        is_corporate=not is_free and not is_disposable

        email_score=1.0
        if is_disposable: email_score-=0.60
        if is_free: email_score-=0.30
        if is_corporate: email_score=min(email_score+0.20,1.0)

        signals=[]
        if is_disposable: signals.append("disposable_email")
        if is_free: signals.append("free_email_provider")
        if is_corporate: signals.append("corporate_email")

        return {
            "emails_found":domains,
            "free_provider":is_free,
            "disposable":is_disposable,
            "corporate_domain":is_corporate,
            "email_score":round(max(0.0,email_score),4),
            "signals":signals,
        }


class WebPresenceExtractor:

    PLATFORM_PATTERNS={
        "linkedin":r"linkedin\.com/company/[\w\-]+",
        "glassdoor":r"glassdoor\.com/(?:overview|reviews)/[\w\-]+",
        "twitter":r"(?:twitter|x)\.com/[\w]+",
        "github":r"github\.com/[\w\-]+",
        "facebook":r"facebook\.com/[\w.\-]+",
        "website":r"https?://(?:www\.)?[\w\-]+\.(?:com|io|co|net|org)/",
        "crunchbase":r"crunchbase\.com/organization/[\w\-]+",
    }

    PLATFORM_MENTION_RE={
        "linkedin":re.compile(r"\blinkedin\b",re.IGNORECASE),
        "glassdoor":re.compile(r"\bglassdoor\b",re.IGNORECASE),
        "twitter":re.compile(r"\b(?:twitter|@\w{3,})\b",re.IGNORECASE),
        "github":re.compile(r"\bgithub\b",re.IGNORECASE),
    }

    def extract(self,text:str)->dict:
        found={}
        for platform,pattern in self.PLATFORM_PATTERNS.items():
            m=re.search(pattern,text,re.IGNORECASE)
            found[platform]=bool(m)

        for platform,pattern in self.PLATFORM_MENTION_RE.items():
            if not found.get(platform):
                found[platform]=bool(pattern.search(text))

        presence_score=sum(1 for v in found.values() if v)/max(len(found),1)

        return {
            "platforms":found,
            "presence_score":round(presence_score,4),
            "count":sum(1 for v in found.values() if v),
        }


class DomainAgeEstimator:

    TRUSTED_OLD_DOMAINS={
        "linkedin.com","indeed.com","glassdoor.com","google.com",
        "microsoft.com","amazon.com","apple.com","ibm.com",
        "oracle.com","salesforce.com","sap.com","cisco.com",
        "accenture.com","deloitte.com","mckinsey.com","pwc.com",
        "infosys.com","tcs.com","wipro.com","cognizant.com",
    }

    def estimate(self,company_name:str,domain:str)->dict:
        domain=domain.lower().strip()

        if any(t in domain for t in self.TRUSTED_OLD_DOMAINS):
            return {
                "age_estimate":"established",
                "age_score":1.0,
                "signal":"known_trusted_domain"
            }

        suspicious=[]
        age_score=0.6

        digit_ratio=sum(c.isdigit() for c in domain.split(".")[0])/max(len(domain.split(".")[0]),1)
        if digit_ratio>0.4:
            suspicious.append("high_digit_ratio_domain")
            age_score-=0.2

        parts=domain.split(".")
        sld=parts[0] if parts else ""
        if len(sld)<4:
            suspicious.append("very_short_domain")
            age_score-=0.1

        tld="."+parts[-1] if parts else ""
        if tld in SUSPICIOUS_TLDS:
            suspicious.append("suspicious_tld")
            age_score-=0.30

        if domain.count("-")>=3:
            suspicious.append("hyphen_stuffed")
            age_score-=0.15

        scam_words={"hire","job","earn","work","cash","income",
                    "salary","pay","money","daily","instant"}

        name_lower=company_name.lower()
        if sum(1 for w in scam_words if w in name_lower)>=2:
            suspicious.append("generic_scam_company_name")
            age_score-=0.20

        age_estimate=(
            "established" if age_score>=0.7 else
            "uncertain" if age_score>=0.4 else
            "suspicious_new"
        )

        return {
            "age_estimate":age_estimate,
            "age_score":round(max(0.0,age_score),4),
            "signal":suspicious if suspicious else ["no_obvious_domain_issues"],
        }


class ContactInfoExtractor:

    PHONE_RE=re.compile(r"\+?[\d\s\-().]{7,15}\d",re.IGNORECASE)

    ADDRESS_RE=re.compile(
        r"\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|blvd|lane|ln|drive|dr|"
        r"court|ct|way|place|pl|suite|floor|building|tower|park|square)",
        re.IGNORECASE
    )

    WHATSAPP_RE=re.compile(r"whatsapp|wa\.me|wapp",re.IGNORECASE)
    TELEGRAM_RE=re.compile(r"telegram|t\.me/@?\w+",re.IGNORECASE)

    def extract(self,text:str)->dict:
        has_phone=bool(self.PHONE_RE.search(text))
        has_address=bool(self.ADDRESS_RE.search(text))
        has_whatsapp=bool(self.WHATSAPP_RE.search(text))
        has_telegram=bool(self.TELEGRAM_RE.search(text))

        contact_score=0.5
        if has_phone: contact_score+=0.20
        if has_address: contact_score+=0.30
        if has_whatsapp: contact_score-=0.20
        if has_telegram: contact_score-=0.20

        signals=[]
        if has_whatsapp: signals.append("whatsapp_only_contact")
        if has_telegram: signals.append("telegram_only_contact")
        if has_address: signals.append("physical_address_present")
        if has_phone: signals.append("phone_number_present")

        return {
            "has_phone":has_phone,
            "has_address":has_address,
            "has_whatsapp":has_whatsapp,
            "has_telegram":has_telegram,
            "contact_score":round(min(max(contact_score,0.0),1.0),4),
            "signals":signals,
        }


class CompanyDescriptionAnalyser:

    VAGUE_PHRASES=[
        "fast growing company","dynamic team","exciting opportunity",
        "no experience needed","work anywhere","be your own boss",
        "we are a leading","reputed company","international company",
        "multinational firm",
    ]

    SPECIFIC_SIGNALS=[
        r"\b(founded|established)\s+in\s+\d{4}",
        r"\b\d+\s+(employees|staff|team\s+members)",
        r"\b(headquartered|offices)\s+in\s+\w+",
        r"\b(revenue|turnover)\s+of\s+[\$₹£€]",
        r"\b(BSE|NSE|NYSE|NASDAQ|listed)\b",
        r"glassdoor\s+rating",
        r"fortune\s+\d+",
    ]

    def analyse(self,text:str)->dict:
        text_lower=text.lower()
        vague_hits=[p for p in self.VAGUE_PHRASES if p in text_lower]

        specific_hits=[
            p for p in self.SPECIFIC_SIGNALS
            if re.search(p,text,re.IGNORECASE)
        ]

        desc_score=0.5
        desc_score-=0.08*len(vague_hits)
        desc_score+=0.15*len(specific_hits)

        word_count=len(text.split())
        if word_count<30: desc_score-=0.20
        if word_count>200: desc_score+=0.10

        return {
            "vague_phrases":vague_hits,
            "specific_signals":specific_hits,
            "word_count":word_count,
            "desc_score":round(min(max(desc_score,0.0),1.0),4),
        }


class CompanyTrustScorer:

    def __init__(self):
        self.email=EmailSignalExtractor()
        self.web=WebPresenceExtractor()
        self.domain=DomainAgeEstimator()
        self.contact=ContactInfoExtractor()
        self.desc=CompanyDescriptionAnalyser()

    def score(self,text:str,
              company_name:str="",
              company_domain:str="")->dict:

        email_r=self.email.extract(text)
        web_r=self.web.extract(text)
        domain_r=self.domain.estimate(company_name,company_domain or company_name)
        contact_r=self.contact.extract(text)
        desc_r=self.desc.analyse(text)

        trust_score=(
            0.25*email_r["email_score"]+
            0.20*web_r["presence_score"]+
            0.20*domain_r["age_score"]+
            0.15*contact_r["contact_score"]+
            0.20*desc_r["desc_score"]
        )

        trust_score=round(min(max(trust_score,0.0),1.0),4)

        fraud_risk=round(1.0-trust_score,4)

        risk_level=(
            "CRITICAL" if fraud_risk>=0.80 else
            "HIGH" if fraud_risk>=0.60 else
            "MEDIUM" if fraud_risk>=0.35 else
            "LOW"
        )

        all_signals=(
            email_r["signals"]+
            web_r.get("signals",[])+
            (domain_r["signal"] if isinstance(domain_r["signal"],list)
             else [domain_r["signal"]])+
            contact_r["signals"]+
            desc_r["vague_phrases"][:2]
        )

        return {
            "trust_score":trust_score,
            "fraud_risk_score":fraud_risk,
            "risk_level":risk_level,
            "breakdown":{
                "email":email_r["email_score"],
                "web_presence":web_r["presence_score"],
                "domain_age":domain_r["age_score"],
                "contact":contact_r["contact_score"],
                "description":desc_r["desc_score"],
            },
            "email":email_r,
            "web_presence":web_r,
            "domain_age":domain_r,
            "contact":contact_r,
            "description":desc_r,
            "signals":all_signals,
            "explanation":self._explain(
                trust_score,
                fraud_risk,
                all_signals,
                email_r,
                web_r
            ),
        }

    @staticmethod
    def _explain(trust,fraud,signals,email_r,web_r)->str:
        if trust>=0.65:
            return f"Company appears credible (trust score: {trust:.0%}). Multiple legitimate signals detected."

        parts=[f"Company trust is LOW (trust: {trust:.0%}, fraud risk: {fraud:.0%})."]

        if email_r.get("free_provider"):
            parts.append("Recruiter uses a free email provider (e.g. Gmail).")

        if email_r.get("disposable"):
            parts.append("Disposable/throwaway email address detected.")

        if web_r["count"]==0:
            parts.append("No professional web presence found (LinkedIn, Glassdoor, etc.).")

        return " ".join(parts)


if __name__=="__main__":
    scorer=CompanyTrustScorer()

    tests=[
        (
            "LEGIT",
            "Acme Corp (founded in 2005, 500+ employees, NASDAQ listed). "
            "Visit our LinkedIn: linkedin.com/company/acme-corp. "
            "Contact: hr@acmecorp.com. Office: 123 Main St, San Francisco CA.",
            "Acme Corp",
            "acmecorp.com"
        ),

        (
            "FAKE",
            "URGENT HIRING!!! Easy Jobs Inc. "
            "WhatsApp: +1-555-0000. Apply now! Registration fee $99. "
            "We are a fast growing company offering guaranteed income. "
            "Contact: easyjobs_hr@gmail.com. Telegram: @easyjobscam",
            "Easy Jobs Inc",
            "easy-jobs-now-earn.tk"
        ),
    ]

    for label,text,name,domain in tests:
        r=scorer.score(text,name,domain)

        print(f"[{label}]")
        print(f"  Trust Score  : {r['trust_score']:.4f}")
        print(f"  Fraud Risk   : {r['fraud_risk_score']:.4f}  [{r['risk_level']}]")
        print(f"  Signals      : {r['signals'][:4]}")
        print(f"  Explanation  : {r['explanation']}")
        print()