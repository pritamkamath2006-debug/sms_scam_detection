from __future__ import annotations

from dataclasses import dataclass


CLASSES: list[str] = ["Safe", "Phishing", "OTP Scam", "Lottery Scam", "Job Scam"]


SEVERITY_BY_CLASS: dict[str, int] = {
    "Safe": 0,
    "Phishing": 90,
    "OTP Scam": 85,
    "Lottery Scam": 75,
    "Job Scam": 70,
}


@dataclass(frozen=True)
class PatternSet:
    keywords: set[str]
    phrases: set[str]


PATTERNS_BY_CLASS: dict[str, PatternSet] = {
    "Phishing": PatternSet(
        keywords={
            "verify",
            "verification",
            "login",
            "password",
            "account",
            "bank",
            "suspend",
            "locked",
            "security",
            "urgent",
            "click",
            "link",
            "reset",
            "update",
        },
        phrases={
            "confirm your account",
            "unusual activity",
            "update your details",
            "reset your password",
            "account will be closed",
            "click the link",
        },
    ),
    "OTP Scam": PatternSet(
        keywords={
            "otp",
            "code",
            "verification code",
            "one time",
            "authenticate",
            "2fa",
            "2-factor",
            "expires",
            "do not share",
        },
        phrases={
            "share the code",
            "send me the otp",
            "verify this code",
            "your code is",
            "expires in",
        },
    ),
    "Lottery Scam": PatternSet(
        keywords={
            "winner",
            "won",
            "prize",
            "lottery",
            "jackpot",
            "claim",
            "free",
            "congratulations",
            "gift",
            "reward",
        },
        phrases={
            "you have won",
            "claim your prize",
            "congratulations you are selected",
            "limited time offer",
        },
    ),
    "Job Scam": PatternSet(
        keywords={
            "job",
            "hiring",
            "recruiter",
            "interview",
            "work from home",
            "remote",
            "salary",
            "pay",
            "training fee",
            "deposit",
            "telegram",
            "whatsapp",
        },
        phrases={
            "pay to start",
            "send your resume",
            "contact our hr",
            "work from home",
            "no experience required",
        },
    ),
}


GLOBAL_SUSPICIOUS_KEYWORDS: set[str] = {
    "urgent",
    "immediately",
    "asap",
    "limited",
    "today",
    "act now",
    "click",
    "link",
    "tap",
    "verify",
    "confirm",
    "gift",
    "reward",
}

