from __future__ import annotations

"""Train the text-only OTP scam baseline when explicitly approved.

This script is intentionally not run during setup.  It validates the derived
dataset before training and only writes artifacts when --save-artifacts is
provided.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# Allow ``python scripts/train_otp_model.py`` to import the sibling package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scamguard.features import build_vectorizer
from scamguard.preprocessing import clean_text


TEXT_COLUMN = "text_transcript"
LABEL_COLUMN = "label"
EXPECTED_LABELS = {"SCAM", "GENUINE"}


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def load_and_validate_dataset(data_path: Path) -> pd.DataFrame:
    """Load the derived CSV and enforce the assumptions of this baseline."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    texts = df[TEXT_COLUMN]
    if texts.isna().any() or texts.astype(str).str.strip().eq("").any():
        raise ValueError(f"{TEXT_COLUMN} contains missing or empty values")

    if texts.duplicated().any():
        duplicate_count = int(texts.duplicated().sum())
        raise ValueError(
            f"{TEXT_COLUMN} contains {duplicate_count} duplicate text row(s). "
            "Deduplicate the derived dataset before training."
        )

    labels = set(df[LABEL_COLUMN].dropna().astype(str))
    unexpected_labels = labels - EXPECTED_LABELS
    missing_labels = EXPECTED_LABELS - labels
    if unexpected_labels or missing_labels:
        raise ValueError(
            "Expected exactly the labels SCAM and GENUINE; "
            f"found {sorted(labels)}"
        )

    if df[LABEL_COLUMN].isna().any():
        raise ValueError(f"{LABEL_COLUMN} contains missing values")

    return df[[TEXT_COLUMN, LABEL_COLUMN]].copy()


def preprocess_texts(texts: pd.Series) -> list[str]:
    """Apply PhishShield's shared minimal text normalization."""
    return [clean_text(text) for text in texts.astype(str)]


def build_classifier() -> LogisticRegression:
    """Return the approved baseline classifier configuration."""
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the OTP Scam text-only TF-IDF + Logistic Regression baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT.parent / "dataset" / "otp_scam_dataset.csv",
        help="Derived CSV containing text_transcript and label columns.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Explicitly save the fitted classifier and TF-IDF vectorizer after training.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("scamguard/artifacts/otp_logistic_regression.joblib"),
    )
    parser.add_argument(
        "--vectorizer-out",
        type=Path,
        default=Path("scamguard/artifacts/otp_tfidf_vectorizer.joblib"),
    )
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1")

    _banner("1. Dataset validation")
    df = load_and_validate_dataset(args.data)
    print(f"Dataset: {args.data}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Class distribution:")
    for label, count in sorted(Counter(df[LABEL_COLUMN]).items()):
        print(f"  {label}: {count}")

    texts = preprocess_texts(df[TEXT_COLUMN])
    labels = df[LABEL_COLUMN].astype(str).tolist()

    _banner("2. Stratified train/test split")
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Test proportion: {args.test_size:.0%}")
    print(f"Random state: {args.seed}")

    _banner("3. TF-IDF + Logistic Regression")
    vectorizer = build_vectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    print(f"TF-IDF features: {x_train_tfidf.shape[1]}")

    classifier = build_classifier()
    fit_started = perf_counter()
    classifier.fit(x_train_tfidf, y_train)
    fit_seconds = perf_counter() - fit_started

    _banner("4. Evaluation from this training run")
    predictions = classifier.predict(x_test_tfidf)
    report_labels = ["GENUINE", "SCAM"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=report_labels,
        zero_division=0,
    )
    print(f"Accuracy: {accuracy_score(y_test, predictions):.12f}")
    print(f"Logistic Regression fit time (seconds): {fit_seconds:.6f}")
    print("Per-class precision, recall, and F1:")
    for label, p, r, score, count in zip(report_labels, precision, recall, f1, support):
        print(
            f"  {label}: precision={p:.12f}, recall={r:.12f}, "
            f"f1={score:.12f}, support={count}"
        )
    print("Confusion matrix (rows=actual, columns=predicted; GENUINE, SCAM):")
    print(confusion_matrix(y_test, predictions, labels=report_labels))
    print(classification_report(y_test, predictions, labels=report_labels, digits=6, zero_division=0))

    if args.save_artifacts:
        _banner("5. Saving approved artifacts")
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        args.vectorizer_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, args.model_out)
        joblib.dump(vectorizer, args.vectorizer_out)
        print(f"Classifier saved to: {args.model_out}")
        print(f"Vectorizer saved to: {args.vectorizer_out}")
    else:
        print("\nArtifacts were not saved. Use --save-artifacts only after approval.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
