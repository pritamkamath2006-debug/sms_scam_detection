from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from scamguard.model import build_model_pipeline, prepare_texts


# ── helpers ──────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    col_w = 14

    # Header row
    header = f"{'':20}" + "".join(f"{lab[:12]:>{col_w}}" for lab in labels)
    print(header)
    print("-" * len(header))

    # Data rows
    for i, row_label in enumerate(labels):
        row = f"{row_label[:18]:20}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(labels)))
        print(row)

    print()
    print("  Rows = Actual label   |   Columns = Predicted label")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train ScamGuard model (TF-IDF + Logistic Regression)."
    )
    parser.add_argument(
        "--data",
        default="data/sample_messages.csv",
        help="CSV with columns: text, label",
    )
    parser.add_argument(
        "--out",
        default="scamguard/artifacts/model.joblib",
        help="Output model path",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _banner("1. Dataset Summary")
    df = pd.read_csv(data_path)
    print(f"  File          : {data_path}")
    print(f"  Total messages: {len(df)}")
    print(f"  Columns       : {list(df.columns)}")
    print()

    label_counts = Counter(df["label"].astype(str))
    print("  Class distribution:")
    for label, count in sorted(label_counts.items()):
        bar = "█" * (count // 2)
        print(f"    {label:<18} {count:>4}  {bar}")

    X = prepare_texts(df["text"].astype(str).tolist())
    y = df["label"].astype(str).tolist()
    labels = sorted(set(y))

    # ── 2. Train / test split ─────────────────────────────────────────────────
    _banner("2. Train / Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test samples  : {len(X_test)}")
    print(f"  Test size     : {args.test_size * 100:.0f}%")
    print(f"  Random seed   : {args.seed}")
    print(f"  Stratified    : Yes (class balance preserved in split)")

    # ── 3. Train model ────────────────────────────────────────────────────────
    _banner("3. Training Model")
    print("  Model         : TF-IDF + Logistic Regression")
    print("  Training...")
    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)
    print("  Done.")

    # ── 4. Evaluation on test set ─────────────────────────────────────────────
    _banner("4. Evaluation on Hold-out Test Set")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Overall Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── 5. Confusion matrix ───────────────────────────────────────────────────
    _banner("5. Confusion Matrix")
    _print_confusion_matrix(y_test, y_pred, labels)

    # ── 6. Cross-validation ───────────────────────────────────────────────────
    _banner(f"6. {args.cv_folds}-Fold Cross-Validation (full dataset)")
    print(f"  Running {args.cv_folds}-fold stratified cross-validation...")
    pipeline_cv = build_model_pipeline()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(pipeline_cv, X, y, cv=cv, scoring="accuracy")

    print(f"\n  Fold scores   : {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"  Mean accuracy : {cv_scores.mean() * 100:.2f}%")
    print(f"  Std deviation : ± {cv_scores.std() * 100:.2f}%")
    print()

    if cv_scores.std() < 0.05:
        print("  ✓ Low variance — model is consistent across folds.")
    else:
        print("  ⚠ High variance — model performance varies. Consider more data.")

    # ── 7. Save model ─────────────────────────────────────────────────────────
    _banner("7. Saving Model")
    joblib.dump(pipeline, out_path)
    print(f"  Saved -> {out_path}")

    # ── 8. Final summary ──────────────────────────────────────────────────────
    _banner("8. Training Complete — Summary")
    print(f"  Dataset       : {len(df)} messages, {len(labels)} classes")
    print(f"  Test accuracy : {acc * 100:.2f}%")
    print(f"  CV accuracy   : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    print(f"  Model saved   : {out_path}")
    print()
    print("  Ready to run: streamlit run app.py")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())