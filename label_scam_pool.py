import pandas as pd
from pathlib import Path
import os

# ============================================================
# SMS SCAM POOL MANUAL LABELING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

INPUT_FILE = DATASET_DIR / "scam_training_pool.csv"
OUTPUT_FILE = DATASET_DIR / "scam_pool_labels.csv"

LABELS = {
    "1": "SCAM",
    "2": "GENUINE",
    "3": "NON-SCAM SPAM",
    "4": "SKIP",
}

# ------------------------------------------------------------
# Load candidate messages
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    print("ERROR: scam_training_pool.csv was not found.")
    print(f"Expected location:")
    print(INPUT_FILE)
    raise SystemExit

df = pd.read_csv(INPUT_FILE)

required_columns = ["message_id", "text", "label"]

for column in required_columns:
    if column not in df.columns:
        print(f"ERROR: Missing column: {column}")
        raise SystemExit

df["message_id"] = df["message_id"].astype(str)
df["text"] = df["text"].fillna("").astype(str)

# ------------------------------------------------------------
# Load previous progress if available
# ------------------------------------------------------------

if OUTPUT_FILE.exists():

    reviewed = pd.read_csv(OUTPUT_FILE)

    if "message_id" not in reviewed.columns:
        print("ERROR: Existing label file is invalid.")
        raise SystemExit

    reviewed["message_id"] = reviewed["message_id"].astype(str)

    already_reviewed = set(reviewed["message_id"])

else:

    reviewed = pd.DataFrame(
        columns=[
            "message_id",
            "text",
            "original_label",
            "manual_label",
        ]
    )

    already_reviewed = set()


# ------------------------------------------------------------
# Find messages that still need review
# ------------------------------------------------------------

remaining = df[
    ~df["message_id"].isin(already_reviewed)
].copy()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

print("=" * 70)
print("SMS SCAM POOL MANUAL LABELING")
print("=" * 70)

print()
print(f"Total candidate messages : {len(df)}")
print(f"Already reviewed         : {len(already_reviewed)}")
print(f"Remaining messages       : {len(remaining)}")

print()
print("LABEL GUIDE")
print("-" * 70)
print("1 = SCAM")
print("    Fraud/deception, fake prize, fake bank/support,")
print("    suspicious payment request, credential theft,")
print("    malicious link, fake delivery, OTP theft, etc.")
print()
print("2 = GENUINE")
print("    Normal personal or legitimate service message.")
print()
print("3 = NON-SCAM SPAM")
print("    Advertisement/promotion without clear fraud.")
print()
print("4 = SKIP")
print("    Unsure or ambiguous.")
print()
print("5 = QUIT")
print("    Save progress and exit.")
print("=" * 70)


# ------------------------------------------------------------
# Nothing left
# ------------------------------------------------------------

if len(remaining) == 0:

    print()
    print("All candidate messages have been reviewed.")
    print()
    print("Final label counts:")

    if len(reviewed) > 0:
        print(reviewed["manual_label"].value_counts())

    raise SystemExit


# ------------------------------------------------------------
# Review messages
# ------------------------------------------------------------

for position, (_, row) in enumerate(remaining.iterrows(), start=1):

    message_id = str(row["message_id"])
    message_text = str(row["text"])
    original_label = str(row["label"])

    print()
    print("=" * 70)
    print(f"REVIEW {position}/{len(remaining)}")
    print("=" * 70)

    print(f"Message ID: {message_id}")
    print(f"Original dataset label: {original_label}")

    print()
    print("SMS:")
    print("-" * 70)
    print(message_text)
    print("-" * 70)

    print()
    print("Choose the correct label:")
    print()
    print("1 = SCAM")
    print("2 = GENUINE")
    print("3 = NON-SCAM SPAM")
    print("4 = SKIP")
    print("5 = QUIT")

    while True:

        choice = input("Enter choice: ").strip()

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if choice == "5":

            print()
            print("Saving progress...")

            reviewed.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8"
            )

            print()
            print("Progress saved:")
            print(OUTPUT_FILE)

            print()
            print("You can run the same command later to continue.")

            raise SystemExit

        # ----------------------------------------------------
        # Valid label
        # ----------------------------------------------------

        if choice in LABELS:

            manual_label = LABELS[choice]

            new_row = pd.DataFrame([{
                "message_id": message_id,
                "text": message_text,
                "original_label": original_label,
                "manual_label": manual_label,
            }])

            reviewed = pd.concat(
                [reviewed, new_row],
                ignore_index=True
            )

            # Save immediately after every answer
            reviewed.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8"
            )

            print()
            print(f"Saved: {manual_label}")

            break

        print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


# ------------------------------------------------------------
# Completed
# ------------------------------------------------------------

print()
print("=" * 70)
print("LABELING SESSION COMPLETED")
print("=" * 70)

print()
print(f"Total labeled in this file: {len(reviewed)}")

print()
print("Current label counts:")

print(reviewed["manual_label"].value_counts())

print()
print("Saved file:")
print(OUTPUT_FILE)

print()
print("You can run the program again later to continue.")