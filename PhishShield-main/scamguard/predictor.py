"""
scamguard/predictor.py
======================
Main inference orchestrator for 3-class SMS Scam Detection.
Integrates text preprocessing, TF-IDF vectorization, model scoring,
rule-based risk indicators, and safety recommendations.
Distinguishes between non-fraudulent commercial marketing (NON-SCAM SPAM)
and fraudulent deception (SCAM).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from .config import CLASSES, SEVERITY_BY_CLASS, CLASS_COLORS, CLASS_DESCRIPTIONS
from .explain import analyze_risk_indicators, compute_composite_risk, ExplainabilityResult
from .features import build_vectorizer
from .model import build_model_pipeline, prepare_texts
from .preprocessing import clean_text
from .tips import safety_tips


@dataclass(frozen=True)
class ScamGuardOutput:
    predicted_class: str
    raw_model_class: str
    confidence_score: float
    confidence_pct: int
    risk_score: int
    risk_level: str
    class_color: str
    class_description: str
    explanation: str
    detected_indicators: list[str]
    matched_snippets: dict[str, list[str]]
    has_url: bool
    has_phone: bool
    has_currency: bool
    tips: list[str]


# Marketing promotional keywords (indicate NON-SCAM SPAM, NOT scam)
PROMO_PATTERNS = [
    r"\bdiscount\b",
    r"\bsale\b",
    r"\boff\b",
    r"\bdeal\b",
    r"\bshop\b",
    r"\bcoupon\b",
    r"\bcollection\b",
    r"\bstore\b",
    r"\bflat\s+\d+%\b",
    r"\bup\s+to\s+\d+%\s+off\b",
    r"\bsubscription\b",
    r"\bcallertune\b",
    r"\bringtone\b",
    r"\bmobile\s+plan\b",
    r"\brecharge\b",
    r"\bcashback\b",
    r"\bhandset\b",
    r"\bbrand\b",
    r"\bopt\s*out\b",
    r"\breply\s+stop\b",
]

# Explicit fraud indicators that must be present for a SCAM classification
CRITICAL_FRAUD_PATTERNS = [
    r"\botp\b",
    r"\bone\s*time\s*password\b",
    r"\bverification\s*code\b",
    r"\bcvv\b",
    r"\bpin(?:\s*code)?\b",
    r"\bpasscode\b",
    r"\bpassword\b",
    r"\bkyc\s+(?:expired|pending|update)\b",
    r"\baccount\s+(?:blocked|suspended|deactivated|locked)\b",
    r"\bnetbanking\b",
    r"\bcard\s+blocked\b",
    r"\bpan\s*card\s+blocked\b",
    r"\b(?:won|winner)\s+(?:a\s+)?(?:lottery|jackpot|cash\s+prize)\b",
    r"\bclaim\s+code\b",
    r"\b0906\d{6,}\b",   # Known UK premium rate fraud prefixes
    r"\b0871\d{6,}\b",
    # Composite fraud: account threat + action link
    r"\b(?:suspended|blocked|deactivated|locked)\b.{0,60}(?:click|visit|update|verify|login|http)",
    r"\b(?:kyc|pan\s*card)\b.{0,60}(?:click|visit|update|verify|login|http)",
    # Phishing pretexts
    r"\bclick\s+here\s+to\s+(?:verify|update|restore|activate|reactivate|unblock)\b",
    r"\bvisit\s+(?:http|www\.)\b",
    r"\bshare\s+(?:your\s+)?(?:otp|card|account|password|pin|cvv|details)\b",
    r"\bdo\s+not\s+share\b.{0,30}\botp\b",  # "do not share your OTP" = OTP context, still SCAM
]

# Patterns that strongly indicate genuine personal or transactional messages
# If these match AND raw model says SCAM AND no critical fraud patterns exist -> override to GENUINE
GENUINE_OVERRIDE_PATTERNS = [
    # Delivery / logistics confirmations (brief)
    r"\bdelivered\b.{0,30}\b(?:door|address|home|flat|building|porch|doorstep|post\s*box|parcel\s*locker)\b",
    r"\bpackage\b.{0,30}\bdelivered\b",
    r"\border\b.{0,30}\b(?:shipped|dispatched|out\s+for\s+delivery|delivered|picked\s+up)\b",
    r"\byour\s+(?:parcel|shipment|courier)\b.{0,30}\bdeliver\b",
    # Short personal "call me" messages — common in genuine conversations
    # Catches: "Call me now", "please call me back", "call me later", "call you later" etc.
    r"\bcall\s+(?:me|you|him|her|us|back|later|now|soon)\b",
    # Uber / Swiggy / food-app arrival
    r"\b(?:uber|ola|rapido|swiggy|zomato|blinkit|dunzo)\b.{0,50}\b(?:arriving|arrived|on\s+the\s+way|delivered|picked\s+up)\b",
]

# Patterns that indicate promotional / commercial spam (NON-SCAM SPAM)
# Applied even when raw model says GENUINE — to catch promotional messages model confuses with genuine
PROMO_RECLASSIFY_PATTERNS = [
    r"\b(?:discount|sale|off|deal|offer)\b.{0,40}\b(?:shop|store|buy|order|visit|available|only|limited)\b",
    r"\b(?:mega|bumper|grand|biggest|huge|exclusive)\s+(?:sale|offer|discount|deal)\b",
    r"\b(?:50|40|30|60|70|80)\s*%\s*(?:off|discount)\b",
    r"\bflat\s+\d+%\s+off\b",
    r"\breply\s+stop\b",
    r"\bopt\s*out\b",
    r"\bsms\s+to\s+stop\b",
    r"\bfashion\s+sale\b",
    r"\bweekend\s+(?:sale|offer|deal)\b",
    r"\b(?:shop|buy)\s+now\b.{0,30}\b(?:discount|sale|off|deal|offer)\b",
]


class ScamGuard:
    def __init__(self, artifacts_dir: str | Path | None = None) -> None:
        if artifacts_dir is None:
            self.artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        else:
            self.artifacts_dir = Path(artifacts_dir)

        self.model_path = self.artifacts_dir / "best_model.joblib"
        self.vectorizer_path = self.artifacts_dir / "tfidf_vectorizer.joblib"

        self._model = None
        self._vectorizer = None

    def ensure_ready(self) -> None:
        """
        Loads the trained model and vectorizer.
        If missing, trains a baseline model on reviewed_dataset.csv.
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self._model is not None and self._vectorizer is not None:
            return

        if self.model_path.exists() and self.vectorizer_path.exists():
            try:
                self._model = joblib.load(self.model_path)
                self._vectorizer = joblib.load(self.vectorizer_path)
                return
            except Exception as e:
                print(f"Error loading existing artifacts: {e}. Will rebuild...")

        self._train_and_save_fallback()

    def _train_and_save_fallback(self) -> None:
        print("[ScamGuard] Training 3-class baseline from reviewed dataset...")
        base_dir = self.artifacts_dir.parents[2]
        data_path = base_dir / "dataset" / "reviewed_dataset.csv"

        if not data_path.exists():
            data_path = base_dir / "dataset" / "manual_labels_reviewed.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {data_path}")

        df = pd.read_csv(data_path)
        label_col = "manual_label" if "manual_label" in df.columns else "label"

        clean_texts = [clean_text(t) for t in df["text"].astype(str)]
        labels = df[label_col].astype(str).str.strip().tolist()

        vectorizer = build_vectorizer(max_features=5000)
        X = vectorizer.fit_transform(clean_texts)

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(X, labels)

        joblib.dump(model, self.model_path)
        joblib.dump(vectorizer, self.vectorizer_path)

        self._model = model
        self._vectorizer = vectorizer
        print(f"[ScamGuard] Saved baseline model to {self.model_path}")

    def analyze(self, raw_message: str) -> ScamGuardOutput:
        """
        Analyzes an incoming SMS message and returns classification and risk breakdown.
        """
        if raw_message is None or not str(raw_message).strip():
            raise ValueError("Input message cannot be empty or whitespace.")

        self.ensure_ready()

        cleaned = clean_text(raw_message)
        X_vec = self._vectorizer.transform([cleaned])

        # Raw statistical model prediction
        raw_model_class = str(self._model.predict(X_vec)[0])

        # Confidence calculation
        confidence = 0.5
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(X_vec)[0]
            confidence = float(np.max(probs))
        elif hasattr(self._model, "decision_function"):
            decision = self._model.decision_function(X_vec)
            if decision.ndim == 1:
                confidence = float(1.0 / (1.0 + np.exp(-abs(decision[0]))))
            else:
                exp_scores = np.exp(decision[0] - np.max(decision[0]))
                probs = exp_scores / np.sum(exp_scores)
                confidence = float(np.max(probs))

        # Rule-based Threat Indicators
        indicators_info = analyze_risk_indicators(raw_message)
        detected_inds = indicators_info["detected_indicators"]

        # Fraud check vs Promo check vs Genuine override:
        has_critical_fraud = any(bool(re.search(pat, cleaned, re.IGNORECASE)) for pat in CRITICAL_FRAUD_PATTERNS)
        has_promo = any(bool(re.search(pat, cleaned, re.IGNORECASE)) for pat in PROMO_PATTERNS)
        has_genuine_signal = any(bool(re.search(pat, cleaned, re.IGNORECASE)) for pat in GENUINE_OVERRIDE_PATTERNS)

        predicted_class = raw_model_class

        # Refinement logic (priority order):
        # 1. Critical fraud signals override everything → SCAM
        if has_critical_fraud:
            predicted_class = "SCAM"
            confidence = max(confidence, 0.88)
        # 2. Model said SCAM but text is a known-genuine delivery/logistics/personal message
        #    with no fraud signals → correct back to GENUINE
        elif raw_model_class == "SCAM" and has_genuine_signal and not has_critical_fraud:
            predicted_class = "GENUINE"
            confidence = 0.72
        # 3. Model said SCAM but message contains commercial promo patterns and no fraud
        elif raw_model_class == "SCAM" and has_promo and not has_critical_fraud:
            predicted_class = "NON-SCAM SPAM"
            confidence = 0.75
        # 4. If raw model class is GENUINE and no fraud patterns: trust model
        elif raw_model_class == "GENUINE" and not has_critical_fraud:
            predicted_class = "GENUINE"

        confidence_pct = int(confidence * 100)

        explain_res = compute_composite_risk(predicted_class, confidence, indicators_info)
        tips = safety_tips(predicted_class)

        return ScamGuardOutput(
            predicted_class=predicted_class,
            raw_model_class=raw_model_class,
            confidence_score=confidence,
            confidence_pct=confidence_pct,
            risk_score=explain_res.composite_risk_score,
            risk_level=explain_res.risk_level,
            class_color=CLASS_COLORS.get(predicted_class, "#666666"),
            class_description=CLASS_DESCRIPTIONS.get(predicted_class, ""),
            explanation=explain_res.explanation,
            detected_indicators=explain_res.detected_indicators,
            matched_snippets=explain_res.matched_snippets,
            has_url=explain_res.has_url,
            has_phone=explain_res.has_phone,
            has_currency=explain_res.has_currency,
            tips=tips,
        )
