import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\Ai based OTP scam detection")

INPUT_FILE = BASE_DIR / "dataset" / "manual_labels.csv"

OUTPUT_FILE = BASE_DIR / "dataset" / "manual_labels_reviewed.csv"


# ============================================================
# LOAD FILE
# ============================================================

if not INPUT_FILE.exists():

    print("ERROR: manual_labels.csv was not found.")
    print(INPUT_FILE)
    raise SystemExit

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "message_id",
    "text",
    "original_label",
    "manual_label"
]

for column in required_columns:

    if column not in df.columns:

        print(f"ERROR: Missing column: {column}")
        raise SystemExit


# ============================================================
# LABEL OPTIONS
# ============================================================

VALID_LABELS = [
    "SCAM",
    "GENUINE",
    "NON-SCAM SPAM"
]


# ============================================================
# FIND MESSAGES THAT NEED REVIEW
# ============================================================

# Review:
# 1. Messages where manual label conflicts with original HAM/SPAM
# 2. Messages that have suspicious manual labels
#
# We DO NOT automatically change any labels.

review_rows = []

for index, row in df.iterrows():

    original = str(
        row["original_label"]
    ).strip().lower()

    manual = str(
        row["manual_label"]
    ).strip().upper()

    text = str(
        row["text"]
    ).strip()

    conflict = False

    # HAM should normally be a genuine/non-scam candidate,
    # but we do not automatically overwrite it.
    if original == "ham" and manual == "SCAM":
        conflict = True

    # SPAM manually marked GENUINE needs review.
    if original == "spam" and manual == "GENUINE":
        conflict = True

    # Empty/unknown labels
    if manual not in VALID_LABELS:
        conflict = True

    if conflict:

        review_rows.append(
            {
                "index": index,
                "message_id": row["message_id"],
                "text": text,
                "original_label": row["original_label"],
                "manual_label": row["manual_label"]
            }
        )


# ============================================================
# REVIEW
# ============================================================

print("=" * 70)
print("SMS MANUAL LABEL REVIEW")
print("=" * 70)

print()

print(
    f"Total manually labeled messages: {len(df)}"
)

print(
    f"Messages requiring review: {len(review_rows)}"
)

print()

if not review_rows:

    print("No conflicts were found.")
    print("No changes were made.")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    raise SystemExit


# ============================================================
# INTERACTIVE REVIEW
# ============================================================

for number, item in enumerate(
    review_rows,
    start=1
):

    index = item["index"]

    print()
    print("=" * 70)
    print(
        f"REVIEW {number}/{len(review_rows)}"
    )
    print("=" * 70)

    print(
        f"Message ID: {item['message_id']}"
    )

    print(
        f"Original dataset label: {item['original_label']}"
    )

    print(
        f"Current manual label: {item['manual_label']}"
    )

    print()
    print("SMS:")
    print("-" * 70)
    print(item["text"])
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

        choice = input(
            "Enter choice: "
        ).strip()

        if choice == "1":

            df.at[
                index,
                "manual_label"
            ] = "SCAM"

            break

        elif choice == "2":

            df.at[
                index,
                "manual_label"
            ] = "GENUINE"

            break

        elif choice == "3":

            df.at[
                index,
                "manual_label"
            ] = "NON-SCAM SPAM"

            break

        elif choice == "4":

            print(
                "Skipped. Existing label kept."
            )

            break

        elif choice == "5":

            print()
            print("Stopping review.")

            df.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print(
                f"Progress saved to:\n{OUTPUT_FILE}"
            )

            raise SystemExit

        else:

            print(
                "Invalid choice. Enter 1, 2, 3, 4 or 5."
            )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("REVIEW COMPLETED")
print("=" * 70)

print()

print(
    df["manual_label"]
    .value_counts()
    .to_string()
)

print()

print(
    f"Saved reviewed file:\n{OUTPUT_FILE}"
)