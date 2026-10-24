from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "dataset" / "sms_scam_categorized.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "sms_scam_validation.csv"

print("=" * 70)
print("SCAM LABEL VALIDATION")
print("=" * 70)

# Load categorized dataset
df = pd.read_csv(INPUT_PATH, encoding="utf-8")

print(f"\nTotal messages: {len(df)}")

# ------------------------------------------------------------
# Create validation columns
# ------------------------------------------------------------

df["final_label"] = ""

df["validation_status"] = "NOT_REVIEWED"

# ------------------------------------------------------------
# Genuine / ham messages
# ------------------------------------------------------------

df.loc[df["label"] == "ham", "final_label"] = "GENUINE"

df.loc[
    df["label"] == "ham",
    "validation_status"
] = "AUTO_GENUINE_FROM_HAM"

# ------------------------------------------------------------
# Spam messages
# ------------------------------------------------------------

spam_mask = df["label"] == "spam"

# Messages detected by our scam indicators
likely_scam_mask = (
    spam_mask &
    (df["likely_scam"] == "YES")
)

df.loc[
    likely_scam_mask,
    "validation_status"
] = "REVIEW_SCAM"

# Spam messages without indicators
no_indicator_mask = (
    spam_mask &
    (df["likely_scam"] == "NO")
)

df.loc[
    no_indicator_mask,
    "validation_status"
] = "REVIEW_OTHER_SPAM"

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print("\nOriginal labels:")
print(df["label"].value_counts())

print("\nValidation status:")
print(df["validation_status"].value_counts())

print("\nLikely scam messages:")
print(
    len(
        df[
            df["validation_status"] == "REVIEW_SCAM"
        ]
    )
)

print("\nOther spam messages:")
print(
    len(
        df[
            df["validation_status"] == "REVIEW_OTHER_SPAM"
        ]
    )
)

# ------------------------------------------------------------
# Show examples for manual review
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LIKELY SCAM EXAMPLES")
print("=" * 70)

scam_review = df[
    df["validation_status"] == "REVIEW_SCAM"
].head(20)

for i, (_, row) in enumerate(
    scam_review.iterrows(),
    start=1
):
    print(f"\n{i}. {row['text']}")
    print(f"   Category: {row['detected_categories']}")

# ------------------------------------------------------------
# Show spam without scam indicators
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OTHER SPAM EXAMPLES")
print("=" * 70)

other_spam = df[
    df["validation_status"] == "REVIEW_OTHER_SPAM"
].head(20)

for i, (_, row) in enumerate(
    other_spam.iterrows(),
    start=1
):
    print(f"\n{i}. {row['text']}")

# ------------------------------------------------------------
# Save validation file
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("VALIDATION FILE CREATED")
print("=" * 70)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print("\nOriginal datasets were NOT modified.")