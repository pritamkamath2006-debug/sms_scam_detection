import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

CLEANED_FILE = DATASET_DIR / "sms_scam_cleaned.csv"
REVIEWED_FILE = DATASET_DIR / "manual_labels_reviewed.csv"
OUTPUT_FILE = DATASET_DIR / "scam_training_pool.csv"

print("=" * 70)
print("PREPARING SCAM-FOCUSED DATASET")
print("=" * 70)

# Load cleaned UCI dataset
df = pd.read_csv(CLEANED_FILE)

# Load manually reviewed labels
reviewed = pd.read_csv(REVIEWED_FILE)

# Normalize columns
reviewed["manual_label"] = reviewed["manual_label"].astype(str).str.upper().str.strip()

# Remove reviewed messages from the automatic pool
reviewed_ids = set(reviewed["message_id"].astype(str))

df["message_id"] = df.index.astype(str)

remaining = df[~df["message_id"].isin(reviewed_ids)].copy()

# Find likely scam messages using conservative indicators
scam_patterns = [
    "urgent",
    "verify",
    "verification",
    "click",
    "link",
    "account suspended",
    "account blocked",
    "winner",
    "won",
    "prize",
    "claim",
    "cash",
    "reward",
    "bank",
    "payment",
    "refund",
    "password",
    "security",
    "otp",
    "pin",
    "credit",
    "debit",
    "loan",
    "delivery",
    "parcel",
    "courier",
    "job offer",
    "selected",
    "fee",
    "money",
]

pattern = "|".join(scam_patterns)

remaining["likely_scam"] = (
    remaining["text"]
    .astype(str)
    .str.lower()
    .str.contains(pattern, regex=True, na=False)
)

pool = remaining[remaining["likely_scam"]].copy()

# Keep useful columns
pool = pool[["message_id", "text", "label"]]

pool.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Original cleaned dataset : {len(df)}")
print(f"Already manually reviewed: {len(reviewed)}")
print(f"Remaining messages       : {len(remaining)}")
print(f"Likely scam candidates   : {len(pool)}")

print()
print("Original label distribution:")
print(pool["label"].value_counts())

print()
print("Saved scam-focused pool:")
print(OUTPUT_FILE)

print()
print("NEXT STEP:")
print("Manually review the scam-focused pool and label:")
print("SCAM / GENUINE / NON-SCAM SPAM")