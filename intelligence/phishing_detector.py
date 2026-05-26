import re
import sys
import math
import json
import string
import hashlib
import pickle
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import (
    SUSPICIOUS_TLDS,
    URL_SHORTENERS,
    PHISHING_KEYWORDS_URL,
    LEGIT_JOB_DOMAINS,
    MODEL_DIR,
    SEED
)

class URLFeatureExtractor:

    IP_RE=re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    HEX_RE=re.compile(r"%[0-9a-fA-F]{2}")
    DIGIT_RE=re.compile(r"\d")

    def extract(self,url:str)->dict:
        url=url.strip()
        parsed=self._safe_parse(url)
        scheme=parsed.scheme.lower()
        host=parsed.netloc.lower().split(":")[0]
        path=parsed.path
        query=parsed.query
        full=url.lower()

        domain_parts=host.split(".")
        tld="."+domain_parts[-1] if len(domain_parts)>1 else ""
        second_level=domain_parts[-2] if len(domain_parts)>=2 else host

        return {
            "url_length":len(url),
            "domain_length":len(host),
            "path_length":len(path),

            "dot_count":url.count("."),
            "hyphen_count":host.count("-"),
            "slash_count":url.count("/"),
            "at_count":url.count("@"),
            "eq_count":url.count("="),
            "question_count":url.count("?"),
            "ampersand_count":url.count("&"),

            "has_https":int(scheme=="https"),
            "has_ip":int(bool(self.IP_RE.match(host))),
            "is_shortener":int(
                host in URL_SHORTENERS or
                any(s in host for s in URL_SHORTENERS)
            ),
            "is_legit_job_site":int(
                any(l in host for l in LEGIT_JOB_DOMAINS)
            ),
            "has_suspicious_tld":int(tld in SUSPICIOUS_TLDS),
            "has_hex_encoding":int(bool(self.HEX_RE.search(url))),
            "has_phishing_kw":int(
                any(k in full for k in PHISHING_KEYWORDS_URL)
            ),
            "subdomain_count":max(0,len(domain_parts)-2),
            "digit_ratio":self._digit_ratio(second_level),

            "domain_entropy":self._entropy(second_level),
        }

    def extract_vector(self,url:str)->list[float]:
        d=self.extract(url)
        return [float(v) for v in d.values()]

    @staticmethod
    def feature_names()->list[str]:
        return [
            "url_length",
            "domain_length",
            "path_length",
            "dot_count",
            "hyphen_count",
            "slash_count",
            "at_count",
            "eq_count",
            "question_count",
            "ampersand_count",
            "has_https",
            "has_ip",
            "is_shortener",
            "is_legit_job_site",
            "has_suspicious_tld",
            "has_hex_encoding",
            "has_phishing_kw",
            "subdomain_count",
            "digit_ratio",
            "domain_entropy",
        ]

    @staticmethod
    def _safe_parse(url:str):
        if not url.startswith(("http://","https://","//")):
            url="http://"+url
        try:
            return urlparse(url)
        except Exception:
            from urllib.parse import ParseResult
            return ParseResult("",
                               url,
                               "",
                               "",
                               "",
                               "")

    @staticmethod
    def _entropy(s:str)->float:
        if not s:
            return 0.0
        freq={c:s.count(c)/len(s) for c in set(s)}
        return -sum(p*math.log2(p) for p in freq.values())

    @staticmethod
    def _digit_ratio(s:str)->float:
        if not s:
            return 0.0
        return sum(c.isdigit() for c in s)/len(s)