"""
train_models.py
===============
Reproducible 3-class model training and comparative evaluation pipeline.
Compares:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
  - Random Forest

Evaluates on a held-out test split with:
  - Accuracy
  - Per-class Precision, Recall, F1
  - Macro-F1
  - Confusion Matrix
Saves best model and vectorizer to scamguard/artifacts/.
"""

import sys
import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Allow importing scamguard
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]  # PhishShield-main
REPO_ROOT = SCRIPT_DIR.parents[1]     # sms_scam_detection

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scamguard.preprocessing import clean_text

DATASET_PATH = REPO_ROOT / "dataset" / "reviewed_dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "scamguard" / "artifacts"
RESULTS_PATH = ARTIFACT_DIR / "model_results.json"
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASSES = ["GENUINE", "NON-SCAM SPAM", "SCAM"]


def load_dataset():
    if not DATASET_PATH.exists():
        fallback = REPO_ROOT / "dataset" / "manual_labels_reviewed.csv"
        if fallback.exists():
            print(f"[Notice] Using fallback dataset: {fallback}")
            df = pd.read_csv(fallback)
        else:
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    else:
        df = pd.read_csv(DATASET_PATH)

    label_col = "manual_label" if "manual_label" in df.columns else "label"
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df[label_col].astype(str).str.strip().str.upper()

    df = df[df["text"] != ""].copy()
    df = df[df["label"].isin(CLASSES)].copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"Loaded {len(df)} validated messages.")
    print("Class distribution:\n", df["label"].value_counts().to_string())
    return df


def main():
    print("=" * 70)
    print("SMS 3-CLASS SCAM DETECTION - MODEL TRAINING & COMPARISON")
    print("=" * 70)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    # Preprocessing
    print("\n[1] Applying text preprocessing...")
    df["cleaned_text"] = [clean_text(t) for t in df["text"]]

    # Stratified Split
    print(f"\n[2] Splitting into 80% train / 20% test (Stratified, random_state={RANDOM_STATE})...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print(f"  Training samples : {len(X_train_raw)}")
    print(f"  Testing samples  : {len(X_test_raw)}")

    # Feature extraction (fit ONLY on training data to avoid data leakage)
    print("\n[3] Fitting TF-IDF Vectorizer on training data...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print(f"  Vocabulary size: {X_train_tfidf.shape[1]}")

    # Model definitions
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    best_model_name = None
    best_macro_f1 = -1.0
    best_scam_recall = -1.0
    best_fitted_model = None

    print("\n[4] Training and evaluating candidate models...\n")

    for name, model in candidate_models.items():
        print("-" * 70)
        print(f"Training: {name}")
        print("-" * 70)

        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        # Per-class metrics
        p_per, r_per, f_per, _ = classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        ), None, None, None

        report_dict = classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred, labels=CLASSES).tolist()

        scam_recall = report_dict["SCAM"]["recall"]
        scam_f1 = report_dict["SCAM"]["f1-score"]

        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "scam_recall": float(scam_recall),
            "scam_f1": float(scam_f1),
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "labels": CLASSES,
        }

        print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Macro-F1 : {macro_f1:.4f}")
        print(f"SCAM Recall: {scam_recall:.4f} | SCAM F1: {scam_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))
        print("Confusion Matrix (Rows=Actual, Columns=Predicted):")
        print(f"Classes: {CLASSES}")
        for r_idx, row in enumerate(cm):
            print(f"  {CLASSES[r_idx]:15}: {row}")
        print()

        # Selection criterion: Highest macro-F1 and SCAM recall
        # We prioritize SCAM recall to minimize dangerous false negatives
        if (macro_f1 > best_macro_f1) or (macro_f1 == best_macro_f1 and scam_recall > best_scam_recall):
            best_macro_f1 = macro_f1
            best_scam_recall = scam_recall
            best_model_name = name
            best_fitted_model = model

    print("=" * 70)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Best Macro-F1      : {best_macro_f1:.4f}")
    print(f"Best SCAM Recall   : {best_scam_recall:.4f}")
    print("=" * 70)

    # Save artifacts
    print(f"\n[5] Saving model artifacts...")
    joblib.dump(best_fitted_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"  Saved model to      : {MODEL_PATH}")
    print(f"  Saved vectorizer to : {VECTORIZER_PATH}")

    # Save evaluation summary
    output_summary = {
        "best_model": best_model_name,
        "best_macro_f1": best_macro_f1,
        "dataset_info": {
            "path": str(DATASET_PATH.relative_to(REPO_ROOT)),
            "total_samples": len(df),
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "classes": CLASSES,
        },
        "models": results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fp:
        json.dump(output_summary, fp, indent=4)
    print(f"  Saved results to    : {RESULTS_PATH}")

    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
