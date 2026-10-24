"""
scamguard/config.py
===================
Configuration for 3-class SMS Scam Detection system.
Defines supported classes, severity weights, and scam indicators.
"""

from __future__ import annotations
from dataclasses import dataclass

# The 3 primary target classes
CLASSES: list[str] = ["GENUINE", "NON-SCAM SPAM", "SCAM"]

# Severity weights used for calculating risk scores (0 to 100)
SEVERITY_BY_CLASS: dict[str, int] = {
    "GENUINE": 5,
    "NON-SCAM SPAM": 35,
    "SCAM": 85,
}

# Color palette for frontend presentation
CLASS_COLORS: dict[str, str] = {
    "GENUINE": "#2e7d32",        # Green
    "NON-SCAM SPAM": "#ed6c02",  # Amber / Orange
    "SCAM": "#d32f2f",           # Red
}

CLASS_DESCRIPTIONS: dict[str, str] = {
    "GENUINE": "Legitimate personal, transactional, or informational communication.",
    "NON-SCAM SPAM": "Unwanted advertisement, promotional marketing, or subscription pitch without clear fraud.",
    "SCAM": "Deceptive or fraudulent message attempting phishing, fake prize claims, OTP/credential theft, or money extortion.",
}


@dataclass(frozen=True)
class PatternSet:
    keywords: set[str]
    phrases: set[str]


# Keyword and phrase patterns for risk analysis
SCAM_INDICATORS: dict[str, list[str]] = {
    "otp_password": [
        r"\botp\b",
        r"\bone\s*time\s*password\b",
        r"\bverification\s*code\b",
        r"\bsecurity\s*code\b",
        r"\bpin(?:\s*code)?\b",
        r"\bpasscode\b",
        r"\bpassword\b",
        r"\bsecret\s*code\b",
    ],
    "account_urgency": [
        r"\bsuspend(?:ed|ing)?\b",
        r"\bblock(?:ed|ing)?\b",
        r"\block(?:ed|ing)?\b",
        r"\bdeactivat(?:ed|ing)?\b",
        r"\baction\s+required\b",
        r"\bexpire(?:d|s)?\b",
        r"\blast\s+chance\b",
        r"\burgent(?:ly)?\b",
        r"\bkyc\b",
        r"\bpan\s*card\b",
        r"\bunusual\s+activity\b",
    ],
    "payment_money": [
        r"\bpay(?:ment)?\b",
        r"\bfee\b",
        r"\btransfer\b",
        r"\brefund\b",
        r"\bcash\b",
        r"\bmoney\b",
        r"\bbill\s*due\b",
        r"\bwire\s*transfer\b",
        r"\bamount\s+due\b",
    ],
    "fake_prize": [
        r"\bwon\b",
        r"\bwinner\b",
        r"\bprize\b",
        r"\blottery\b",
        r"\bclaim\b",
        r"\bjackpot\b",
        r"\breward\b",
        r"\bcongratulations\b",
        r"\bselected\s+to\s+receive\b",
    ],
    "suspicious_link": [
        r"https?://\S+",
        r"www\.\S+",
        r"bit\.ly/\S+",
        r"tinyurl\.com/\S+",
        r"\.(?:xyz|top|info|tk|biz|online|site|ru|cc)\b",
    ],
    "confidential_data": [
        r"\bcard\s*number\b",
        r"\bcvv\b",
        r"\bcredential(?:s)?\b",
        r"\baadhaar\b",
        r"\bbank\s*details\b",
        r"\blogin\b",
        r"\bnetbanking\b",
    ],
}
