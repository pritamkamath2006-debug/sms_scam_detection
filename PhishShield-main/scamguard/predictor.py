"""
scamguard/predictor.py
======================
Main inference orchestrator for 3-class SMS Scam Detection.
Integrates text preprocessing, TF-IDF vectorization, model scoring,
rule-based risk indicators, and safety recommendations.
"""

from __future__ import annotations
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
        If missing, trains a baseline LinearSVC model on reviewed_dataset.csv.
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

        # Fallback: Train automatically on reviewed dataset
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

        vectorizer = build_vectorizer(max_features=10000)
        X = vectorizer.fit_transform(clean_texts)

        from sklearn.svm import LinearSVC
        model = LinearSVC(class_weight="balanced", random_state=42)
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

        # Fraud heuristic check:
        # If the statistical model was trained on historical data lacking modern scam patterns,
        # detect high-risk combinations:
        # 1. Direct OTP / Password request
        # 2. Account block/urgency + web link
        # 3. Fake prize claims with call/claim action
        has_otp_threat = any("otp" in ind.lower() for ind in detected_inds)
        has_urgency = any("urgency" in ind.lower() or "suspension" in ind.lower() for ind in detected_inds)
        has_prize = any("prize" in ind.lower() or "lottery" in ind.lower() for ind in detected_inds)
        has_url = indicators_info["has_url"]

        predicted_class = raw_model_class

        if has_otp_threat:
            predicted_class = "SCAM"
            confidence = max(confidence, 0.85)
        elif has_urgency and has_url:
            predicted_class = "SCAM"
            confidence = max(confidence, 0.90)
        elif has_prize and (has_urgency or indicators_info["has_phone"] or has_url):
            predicted_class = "SCAM"
            confidence = max(confidence, 0.88)

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
