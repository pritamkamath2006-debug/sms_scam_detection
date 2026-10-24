"""
scamguard/explain.py
====================
Explainability and transparent risk-analysis layer for SMS Scam Detection.
Analyzes text for concrete threat patterns:
  1. Requests for OTPs or passwords
  2. Urgent account-blocking threats
  3. Suspicious payment requests
  4. Fake prizes or lottery claims
  5. Suspicious links
  6. Requests for confidential information
Keeps rule-based indicators clearly separated from ML classification.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from .config import SCAM_INDICATORS
from .preprocessing import clean_text

_URL_RE = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,5}[\s-]?\d{3,5}")
_CURRENCY_RE = re.compile(r"[$₹€£]\s?\d+|\b\d+(?:\.\d{1,2})?\s?(?:usd|inr|eur|gbp|pounds|rs)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ExplainabilityResult:
    detected_indicators: list[str]
    matched_snippets: dict[str, list[str]]
    has_url: bool
    has_phone: bool
    has_currency: bool
    indicator_risk_score: int
    risk_level: str
    composite_risk_score: int
    explanation: str


INDICATOR_DESCRIPTIONS: dict[str, str] = {
    "otp_password": "Requests for OTP, password, or security PIN",
    "account_urgency": "Urgent account suspension, block, or expiration threat",
    "payment_money": "Suspicious payment, money transfer, or fee request",
    "fake_prize": "Fake prize, reward, or lottery claim",
    "suspicious_link": "Suspicious web link or URL",
    "confidential_data": "Request for confidential personal or financial data",
}

INDICATOR_WEIGHTS: dict[str, int] = {
    "otp_password": 25,
    "account_urgency": 20,
    "payment_money": 18,
    "fake_prize": 20,
    "suspicious_link": 18,
    "confidential_data": 25,
}


def analyze_risk_indicators(raw_text: str) -> dict:
    """
    Scans SMS text against rule-based risk patterns.
    Does NOT treat a single keyword as conclusive proof of fraud.
    """
    text = clean_text(raw_text)

    has_url = bool(_URL_RE.search(text))
    has_phone = bool(_PHONE_RE.search(text))
    has_currency = bool(_CURRENCY_RE.search(text))

    detected_indicators: list[str] = []
    matched_snippets: dict[str, list[str]] = {}
    raw_indicator_score = 0

    for key, patterns in SCAM_INDICATORS.items():
        matches = []
        for pat in patterns:
            found = re.findall(pat, text, re.IGNORECASE)
            if found:
                matches.extend(found)
        if matches:
            label = INDICATOR_DESCRIPTIONS.get(key, key)
            detected_indicators.append(label)
            matched_snippets[label] = list(set(matches))
            raw_indicator_score += INDICATOR_WEIGHTS.get(key, 10)

    # Add URL indicator if detected by regex
    if has_url and INDICATOR_DESCRIPTIONS["suspicious_link"] not in detected_indicators:
        desc = INDICATOR_DESCRIPTIONS["suspicious_link"]
        detected_indicators.append(desc)
        matched_snippets[desc] = ["web link"]
        raw_indicator_score += INDICATOR_WEIGHTS["suspicious_link"]

    indicator_score = min(50, raw_indicator_score)

    return {
        "detected_indicators": detected_indicators,
        "matched_snippets": matched_snippets,
        "has_url": has_url,
        "has_phone": has_phone,
        "has_currency": has_currency,
        "indicator_score": indicator_score,
    }


def compute_composite_risk(
    predicted_class: str,
    confidence: float,
    indicators_info: dict
) -> ExplainabilityResult:
    """
    Combines the ML prediction with rule-based risk indicators
    into a calibrated transparent score (0 to 100).
    """
    detected_indicators = indicators_info["detected_indicators"]
    matched_snippets = indicators_info["matched_snippets"]
    has_url = indicators_info["has_url"]
    has_phone = indicators_info["has_phone"]
    has_currency = indicators_info["has_currency"]
    ind_score = indicators_info["indicator_score"]

    # Base risk score from ML model
    if predicted_class == "SCAM":
        base_score = 65 + int(confidence * 15)  # 65 - 80
        total_score = min(100, base_score + int(ind_score * 0.4))
    elif predicted_class == "NON-SCAM SPAM":
        base_score = 30 + int(confidence * 15)  # 30 - 45
        total_score = min(75, base_score + int(ind_score * 0.3))
    else:  # GENUINE
        base_score = 5 + int((1.0 - confidence) * 15)  # 5 - 20
        # If genuine model prediction has suspicious indicators, reflect in risk
        total_score = min(60, base_score + int(ind_score * 0.5))

    # Risk level classification
    if total_score >= 75:
        risk_level = "CRITICAL / HIGH RISK"
    elif total_score >= 45:
        risk_level = "MODERATE RISK"
    elif total_score >= 20:
        risk_level = "LOW RISK"
    else:
        risk_level = "VERY LOW / SAFE"

    # Explanation text
    reasons = []
    if detected_indicators:
        reasons.append("Threat indicators observed: " + ", ".join(detected_indicators))
    if has_url:
        reasons.append("Contains clickable link/URL")
    if has_currency:
        reasons.append("Mentions monetary values or payments")

    if predicted_class == "SCAM":
        exp = f"Classified as SCAM by NLP model. " + ("; ".join(reasons) if reasons else "Language patterns strongly match known fraudulent campaigns.")
    elif predicted_class == "NON-SCAM SPAM":
        exp = f"Classified as NON-SCAM SPAM. Appears to be commercial or promotional marketing without conclusive signs of fraud."
    else:
        if detected_indicators:
            exp = f"Classified as GENUINE by model, but caution is recommended due to: " + "; ".join(detected_indicators) + "."
        else:
            exp = "Classified as GENUINE. Consistent with legitimate personal, transactional, or informational communication."

    return ExplainabilityResult(
        detected_indicators=detected_indicators,
        matched_snippets=matched_snippets,
        has_url=has_url,
        has_phone=has_phone,
        has_currency=has_currency,
        indicator_risk_score=ind_score,
        risk_level=risk_level,
        composite_risk_score=total_score,
        explanation=exp,
    )
