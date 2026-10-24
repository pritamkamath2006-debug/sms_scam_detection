from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import CLASSES, SEVERITY_BY_CLASS
from .explain import detect_patterns, pattern_risk_score
from .model import build_model_pipeline, prepare_texts
from .preprocessing import clean_text
from .tips import safety_tips


@dataclass(frozen=True)
class ScamGuardOutput:
    label: str
    probability: float
    confidence_pct: int
    risk_score: int
    reason: str
    suspicious_keywords: list[str]
    suspicious_phrases: list[str]
    tips: list[str]


class ScamGuard:

    def __init__(
        self,
        artifacts_dir: str | Path = "scamguard/artifacts"
    ) -> None:

        self.artifacts_dir = Path(artifacts_dir)

        # New ML model files
        self.model_path = self.artifacts_dir / "best_model.joblib"
        self.vectorizer_path = (
            self.artifacts_dir / "tfidf_vectorizer.joblib"
        )

        # Old pipeline file - kept as fallback
        self.old_model_path = (
            self.artifacts_dir / "model.joblib"
        )

        self._model = None
        self._vectorizer = None
        self._pipeline = None

    # ============================================================
    # LOAD MODEL
    # ============================================================

    def ensure_ready(self) -> None:

        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------------
        # FIRST: Load the new trained model
        # --------------------------------------------------------

        if (
            self._model is not None
            and self._vectorizer is not None
        ):
            return

        if (
            self.model_path.exists()
            and self.vectorizer_path.exists()
        ):

            print("Loading trained ML model...")

            self._model = joblib.load(
                self.model_path
            )

            self._vectorizer = joblib.load(
                self.vectorizer_path
            )

            print("Trained Linear SVM model loaded.")

            return

        # --------------------------------------------------------
        # FALLBACK: Load old model.joblib
        # --------------------------------------------------------

        if self.old_model_path.exists():

            print(
                "New model not found."
                " Loading existing model.joblib..."
            )

            self._pipeline = joblib.load(
                self.old_model_path
            )

            return

        # --------------------------------------------------------
        # LAST FALLBACK: Train old starter model
        # --------------------------------------------------------

        print(
            "No trained model found."
            " Creating starter model..."
        )

        data_path = Path(
            "data/sample_messages.csv"
        )

        if not data_path.exists():

            raise FileNotFoundError(
                "No trained model found and starter "
                "dataset is missing at "
                "data/sample_messages.csv"
            )

        df = pd.read_csv(data_path)

        texts = prepare_texts(
            df["text"].astype(str).tolist()
        )

        labels = (
            df["label"]
            .astype(str)
            .tolist()
        )

        pipeline = build_model_pipeline()

        pipeline.fit(
            texts,
            labels
        )

        joblib.dump(
            pipeline,
            self.old_model_path
        )

        self._pipeline = pipeline

    # ============================================================
    # ANALYZE SMS
    # ============================================================

    def analyze(
        self,
        message: str
    ) -> ScamGuardOutput:

        self.ensure_ready()

        # Clean message
        cleaned = clean_text(message)

        # ========================================================
        # NEW MODEL: Linear SVM + TF-IDF
        # ========================================================

        if (
            self._model is not None
            and self._vectorizer is not None
        ):

            # Convert message to TF-IDF
            features = self._vectorizer.transform(
                [cleaned]
            )

            # Predict class
            prediction = self._model.predict(
                features
            )[0]

            # Convert numeric prediction to label
            if int(prediction) == 1:
                label = "SPAM"
            else:
                label = "HAM"

            # ----------------------------------------------------
            # LinearSVC does not provide predict_proba().
            # Therefore use decision_function as a
            # confidence-like score.
            # ----------------------------------------------------

            decision_score = float(
                self._model.decision_function(
                    features
                )[0]
            )

            # Convert decision score into 0-1 range.
            probability = (
                abs(decision_score)
                / (1.0 + abs(decision_score))
            )

            confidence_pct = int(
                round(probability * 100)
            )

        # ========================================================
        # OLD MODEL FALLBACK
        # ========================================================

        elif self._pipeline is not None:

            proba = self._pipeline.predict_proba(
                [cleaned]
            )[0]

            classes = list(
                self._pipeline
                .named_steps["clf"]
                .classes_
            )

            best_idx = int(
                proba.argmax()
            )

            label = classes[best_idx]

            probability = float(
                proba[best_idx]
            )

            confidence_pct = int(
                round(probability * 100)
            )

        else:

            raise RuntimeError(
                "No ML model is loaded."
            )

        # ========================================================
        # EXPLANATION
        # ========================================================

        expl = detect_patterns(
            message,
            label
        )

        # ========================================================
        # RISK SCORE
        # ========================================================

        severity = SEVERITY_BY_CLASS.get(
            label,
            50
        )

        base_risk = int(
            round(
                severity * probability
            )
        )

        risk = min(
            100,
            base_risk
            + pattern_risk_score(expl)
        )

        # If predicted safe but suspicious
        # patterns exist, avoid showing 0 risk.
        if (
            label in ["HAM", "Safe"]
            and risk == 0
            and (
                expl.suspicious_keywords
                or expl.suspicious_phrases
            )
        ):
            risk = 10

        # ========================================================
        # RETURN RESULT
        # ========================================================

        return ScamGuardOutput(

            label=(
                label
                if label in CLASSES
                else str(label)
            ),

            probability=probability,

            confidence_pct=confidence_pct,

            risk_score=risk,

            reason=expl.reason,

            suspicious_keywords=(
                expl.suspicious_keywords
            ),

            suspicious_phrases=(
                expl.suspicious_phrases
            ),

            tips=safety_tips(label)
        )


# ================================================================
# SIMPLE TEST
# ================================================================

if __name__ == "__main__":

    guard = ScamGuard()

    test_messages = [

        (
            "Congratulations! You have won "
            "a cash prize. Call now."
        ),

        (
            "Hey bro, are you coming to "
            "college today?"
        ),

        (
            "URGENT! You have won a £900 prize. "
            "Call 09061701939 now."
        )
    ]

    print("\n")
    print("=" * 70)
    print("SMS MODEL TEST")
    print("=" * 70)

    for message in test_messages:

        result = guard.analyze(
            message
        )

        print("\nMessage:")
        print(message)

        print(
            "Prediction:",
            result.label
        )

        print(
            "Confidence:",
            result.confidence_pct,
            "%"
        )

        print(
            "Risk Score:",
            result.risk_score
        )

        print(
            "Reason:",
            result.reason
        )

    print("\n")
    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)