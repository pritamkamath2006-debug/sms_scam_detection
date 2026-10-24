from __future__ import annotations

import re
from dataclasses import dataclass

from .config import GLOBAL_SUSPICIOUS_KEYWORDS, PATTERNS_BY_CLASS
from .preprocessing import clean_text

_URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b")
_MONEY_RE = re.compile(r"[$₹€£]\s?\d+|\b\d+(?:\.\d{1,2})?\s?(?:usd|inr|eur|gbp)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ExplainabilityResult:
    suspicious_keywords: list[str]
    suspicious_phrases: list[str]
    has_url: bool
    has_phone: bool
    has_money: bool
    reason: str


def detect_patterns(raw_text: str, predicted_label: str) -> ExplainabilityResult:
    text = clean_text(raw_text)

    has_url = bool(_URL_RE.search(text))
    has_phone = bool(_PHONE_RE.search(text))
    has_money = bool(_MONEY_RE.search(text))

    tokens = set(re.findall(r"[a-z0-9']+", text))
    suspicious_keywords: set[str] = set()
    suspicious_phrases: set[str] = set()

    # Global keywords
    for kw in GLOBAL_SUSPICIOUS_KEYWORDS:
        if kw in text:
            suspicious_keywords.add(kw)

    # Class-specific patterns
    patterns = PATTERNS_BY_CLASS.get(predicted_label)
    if patterns:
        for kw in patterns.keywords:
            if kw in text or kw in tokens:
                suspicious_keywords.add(kw)
        for phrase in patterns.phrases:
            if phrase in text:
                suspicious_phrases.add(phrase)

    # Build reason text
    parts: list[str] = []
    if suspicious_phrases:
        parts.append("Scam-style phrases detected")
    if suspicious_keywords:
        parts.append("Suspicious keywords detected")
    if has_url:
        parts.append("Contains a link")
    if has_phone:
        parts.append("Contains a phone number")
    if has_money:
        parts.append("Mentions money or payment")

    if predicted_label == "Safe" and not parts:
        reason = "No common scam patterns detected."
    else:
        reason = "; ".join(parts) if parts else "Patterns consistent with this scam category."

    return ExplainabilityResult(
        suspicious_keywords=sorted(suspicious_keywords),
        suspicious_phrases=sorted(suspicious_phrases),
        has_url=has_url,
        has_phone=has_phone,
        has_money=has_money,
        reason=reason,
    )


def pattern_risk_score(expl: ExplainabilityResult) -> int:
    score = 0
    score += 6 * len(expl.suspicious_keywords)
    score += 10 * len(expl.suspicious_phrases)
    if expl.has_url:
        score += 10
    if expl.has_phone:
        score += 6
    if expl.has_money:
        score += 10
    return min(40, score)

