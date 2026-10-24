import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

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
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR.parent
    / "dataset"
    / "sms_scam_cleaned.csv"
)

ARTIFACT_DIR = BASE_DIR / "scamguard" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = ARTIFACT_DIR / "model_results.json"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

MAX_FEATURES = 15000
NGRAM_RANGE = (1, 2)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("SMS SCAM DETECTION - MODEL TRAINING")
print("=" * 70)

print("\n[1] Loading dataset...")
print(f"Dataset: {DATASET_PATH}")

if not DATASET_PATH.exists():
    print("\nERROR: Dataset file not found!")
    print(f"Expected path:\n{DATASET_PATH}")
    sys.exit(1)

df = pd.read_csv(DATASET_PATH)

print(f"Rows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "text" not in df.columns or "label" not in df.columns:
    print("\nERROR: Dataset must contain:")
    print("  text")
    print("  label")
    print("\nAvailable columns:")
    print(df.columns.tolist())
    sys.exit(1)


# ============================================================
# CLEAN DATA
# ============================================================

print("\n[2] Cleaning data...")

df = df[["text", "label"]].copy()

df["text"] = df["text"].fillna("").astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.upper()

df = df[df["text"] != ""]

df = df.drop_duplicates(subset=["text"])

print(f"Rows after cleaning: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# LABEL ENCODING
# ============================================================

print("\n[3] Preparing labels...")

# UCI dataset:
# HAM  -> 0
# SPAM -> 1

label_map = {
    "HAM": 0,
    "SPAM": 1
}

unknown_labels = set(df["label"]) - set(label_map.keys())

if unknown_labels:
    print("\nERROR: Unknown labels found:")
    print(unknown_labels)
    sys.exit(1)

df["target"] = df["label"].map(label_map)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[4] Splitting dataset...")

X = df["text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# TF-IDF
# ============================================================

print("\n[5] Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=NGRAM_RANGE,
    max_features=MAX_FEATURES,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF training shape: {X_train_tfidf.shape}")
print(f"TF-IDF testing shape : {X_test_tfidf.shape}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = {}

best_model_name = None
best_model = None
best_f1 = -1


print("\n[6] Training models...")
print("=" * 70)


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    # Train
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    # Store results
    results[model_name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }

    # Print
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["HAM", "SPAM"],
            zero_division=0
        )
    )

    # Select best model using F1
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name
        best_model = model


# ============================================================
# SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("[7] Saving best model...")
print("=" * 70)

print(f"Best model: {best_model_name}")
print(f"Best F1   : {best_f1:.4f}")


MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"

joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "best_model": best_model_name,
    "best_f1_score": float(best_f1),

    "dataset": {
        "file": str(DATASET_PATH),
        "total_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test))
    },

    "label_mapping": {
        "HAM": 0,
        "SPAM": 1
    },

    "tfidf": {
        "ngram_range": list(NGRAM_RANGE),
        "max_features": MAX_FEATURES,
        "sublinear_tf": True
    },

    "random_state": RANDOM_STATE,

    "models": results
}


with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=4)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison_rows = []

for name, result in results.items():

    comparison_rows.append({
        "Model": name,
        "Accuracy": result["accuracy"],
        "Precision": result["precision"],
        "Recall": result["recall"],
        "F1 Score": result["f1_score"]
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.sort_values(
    by="F1 Score",
    ascending=False
)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"\nBest model       : {best_model_name}")
print(f"Best F1 Score    : {best_f1:.4f}")

print(f"\nSaved model:")
print(MODEL_PATH)

print(f"\nSaved TF-IDF:")
print(VECTORIZER_PATH)

print(f"\nSaved results:")
print(RESULTS_PATH)

print("\nNOTE:")
print("This model is trained on the UCI HAM/SPAM dataset.")
print("Therefore, these metrics represent a SPAM classification baseline,")
print("not a final SCAM-vs-GENUINE ground-truth evaluation.")