"""
validate_datasets.py
====================
Inspects and validates all SMS scam detection datasets.
Checks for missing values, duplicates, label consistency,
and generates dataset summaries.
"""

import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

VALID_LABELS = {"SCAM", "GENUINE", "NON-SCAM SPAM"}


def create_consolidated_reviewed_dataset():
    """
    Consolidates verified human labels from manual_labels_reviewed.csv
    and scam_pool_labels.csv without fabricating any data.
    """
    rev_path = DATASET_DIR / "manual_labels_reviewed.csv"
    pool_lbl_path = DATASET_DIR / "scam_pool_labels.csv"
    output_path = DATASET_DIR / "reviewed_dataset.csv"

    records = []

    if rev_path.exists():
        df_rev = pd.read_csv(rev_path)
        for _, row in df_rev.iterrows():
            text = str(row.get("text", "")).strip()
            label = str(row.get("manual_label", "")).strip().upper()
            orig = str(row.get("original_label", "")).strip()
            if text and label in VALID_LABELS:
                records.append({
                    "text": text,
                    "manual_label": label,
                    "original_label": orig,
                    "source": "manual_labels_reviewed"
                })

    if pool_lbl_path.exists():
        df_pool = pd.read_csv(pool_lbl_path)
        for _, row in df_pool.iterrows():
            text = str(row.get("text", "")).strip()
            label = str(row.get("manual_label", "")).strip().upper()
            orig = str(row.get("original_label", "")).strip()
            if text and label in VALID_LABELS:
                records.append({
                    "text": text,
                    "manual_label": label,
                    "original_label": orig,
                    "source": "scam_pool_labels"
                })

    df = pd.DataFrame(records)
    # Deduplicate by text, keeping first (reviewed)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["message_id"] = range(1, len(df) + 1)
    df = df[["message_id", "text", "original_label", "manual_label", "source"]]

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Consolidated reviewed dataset saved to: {output_path} ({len(df)} rows)")
    return df


def validate_all_datasets():
    """
    Validates all relevant dataset files in dataset/.
    """
    print("=" * 70)
    print("DATASET VALIDATION AUDIT")
    print("=" * 70)

    # 1. Consolidated reviewed dataset
    reviewed_df = create_consolidated_reviewed_dataset()
    print("\n--- Consolidated Reviewed Dataset (Gold Standard) ---")
    print(f"Total Rows: {len(reviewed_df)}")
    print(f"Missing Texts: {reviewed_df['text'].isna().sum()}")
    print(f"Duplicate Texts: {reviewed_df['text'].duplicated().sum()}")
    print("Class Distribution:")
    print(reviewed_df["manual_label"].value_counts())

    # 2. Candidate pool
    pool_path = DATASET_DIR / "scam_training_pool.csv"
    if pool_path.exists():
        df_pool = pd.read_csv(pool_path)
        print("\n--- Candidate Review Pool (scam_training_pool.csv) ---")
        print(f"Total Candidate Messages: {len(df_pool)}")
        print(f"Missing Texts: {df_pool['text'].isna().sum()}")
        print(f"Duplicate Texts: {df_pool['text'].duplicated().sum()}")

        reviewed_texts = set(reviewed_df["text"].str.strip().str.lower())
        pool_texts = set(df_pool["text"].str.strip().str.lower())
        already_done = len(reviewed_texts.intersection(pool_texts))
        print(f"Candidates already reviewed: {already_done}")
        print(f"Candidates remaining for review: {len(df_pool) - already_done}")

    # 3. Cleaned UCI dataset
    clean_path = DATASET_DIR / "sms_scam_cleaned.csv"
    if clean_path.exists():
        df_clean = pd.read_csv(clean_path)
        print("\n--- Cleaned UCI SMS Dataset (sms_scam_cleaned.csv) ---")
        print(f"Total Cleaned Messages: {len(df_clean)}")
        print(f"Missing Texts: {df_clean['text'].isna().sum()}")
        print(f"Duplicate Texts: {df_clean['text'].duplicated().sum()}")
        print("Original Distribution:")
        print(df_clean["label"].value_counts())

    print("\n" + "=" * 70)
    print("DATASET VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    validate_all_datasets()
