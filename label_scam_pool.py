"""
label_scam_pool.py
==================
Interactive CLI tool for manual labeling of candidate SMS messages.
Supports 3-class annotations: SCAM, GENUINE, NON-SCAM SPAM.
Allows SKIP (uncertain) and safe checkpoint saving.
Deduplicates based on normalized text content.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

INPUT_FILE = DATASET_DIR / "scam_training_pool.csv"
OUTPUT_FILE = DATASET_DIR / "scam_pool_labels.csv"
REVIEWED_FILE = DATASET_DIR / "reviewed_dataset.csv"

LABELS = {
    "1": "SCAM",
    "2": "GENUINE",
    "3": "NON-SCAM SPAM",
    "4": "SKIP",
}


def load_candidate_pool():
    if not INPUT_FILE.exists():
        print(f"ERROR: Candidate pool file not found at {INPUT_FILE}")
        raise SystemExit(1)
    df = pd.read_csv(INPUT_FILE)
    df["message_id"] = df.get("message_id", range(1, len(df) + 1)).astype(str)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    return df[df["text"] != ""].copy()


def load_already_reviewed_texts():
    reviewed_texts = set()

    # Load from reviewed_dataset.csv if exists
    if REVIEWED_FILE.exists():
        try:
            rev_df = pd.read_csv(REVIEWED_FILE)
            for t in rev_df["text"].dropna():
                reviewed_texts.add(t.strip().lower())
        except Exception as e:
            print(f"Warning loading reviewed_dataset.csv: {e}")

    # Load from scam_pool_labels.csv if exists
    if OUTPUT_FILE.exists():
        try:
            pool_df = pd.read_csv(OUTPUT_FILE)
            for t in pool_df["text"].dropna():
                reviewed_texts.add(t.strip().lower())
        except Exception as e:
            print(f"Warning loading scam_pool_labels.csv: {e}")

    return reviewed_texts


def load_pool_labels_df():
    if OUTPUT_FILE.exists():
        try:
            return pd.read_csv(OUTPUT_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["message_id", "text", "original_label", "manual_label"])


def main():
    df_pool = load_candidate_pool()
    already_reviewed_texts = load_already_reviewed_texts()
    reviewed_df = load_pool_labels_df()

    # Filter out messages whose text is already reviewed
    remaining = df_pool[
        ~df_pool["text"].str.strip().str.lower().isin(already_reviewed_texts)
    ].copy().reset_index(drop=True)

    print("=" * 70)
    print("SMS SCAM POOL MANUAL LABELING")
    print("=" * 70)
    print(f"Total candidate messages : {len(df_pool)}")
    print(f"Already reviewed texts   : {len(already_reviewed_texts)}")
    print(f"Remaining to review      : {len(remaining)}")
    print()
    print("LABEL GUIDE:")
    print("  1 = SCAM           (Phishing, fake prize, bank alert, OTP theft, fraud)")
    print("  2 = GENUINE        (Personal message, valid notification, legitimate service)")
    print("  3 = NON-SCAM SPAM  (Promotional advert, marketing, ringtone offer without fraud)")
    print("  4 = SKIP           (Uncertain, ambiguous, or needs discussion)")
    print("  5 = QUIT           (Save progress safely and exit)")
    print("=" * 70)

    if len(remaining) == 0:
        print("\nAll candidate messages have already been reviewed!")
        print("Label counts in scam_pool_labels.csv:")
        print(reviewed_df["manual_label"].value_counts())
        return

    new_rows = []

    for position, (_, row) in enumerate(remaining.iterrows(), start=1):
        message_id = str(row["message_id"])
        message_text = str(row["text"])
        original_label = str(row.get("label", ""))

        print()
        print("=" * 70)
        print(f"MESSAGE {position} of {len(remaining)}")
        print("=" * 70)
        print(f"Message ID : {message_id}")
        print(f"Orig Label : {original_label}")
        print("\nSMS TEXT:")
        print("-" * 70)
        print(message_text)
        print("-" * 70)
        print("Options: [1] SCAM  [2] GENUINE  [3] NON-SCAM SPAM  [4] SKIP  [5] QUIT")

        while True:
            choice = input("Enter choice (1-5): ").strip()

            if choice == "5":
                print("\nSaving progress and exiting...")
                if new_rows:
                    append_df = pd.DataFrame(new_rows)
                    combined = pd.concat([reviewed_df, append_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["text"], keep="last")
                    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
                    print(f"Saved {len(new_rows)} new labels to: {OUTPUT_FILE}")
                else:
                    print("No new labels added.")
                return

            elif choice in LABELS:
                chosen_label = LABELS[choice]
                new_rows.append({
                    "message_id": message_id,
                    "text": message_text,
                    "original_label": original_label,
                    "manual_label": chosen_label,
                })
                print(f" -> Recorded as: {chosen_label}")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")

    # If completed all
    if new_rows:
        append_df = pd.DataFrame(new_rows)
        combined = pd.concat([reviewed_df, append_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"], keep="last")
        combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\nCompleted! Saved all labels to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
