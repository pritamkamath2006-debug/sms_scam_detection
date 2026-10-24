import pandas as pd
from pathlib import Path

# Original project dataset
SOURCE = Path(
    r"D:\Ai based OTP scam detection\PhishShield-main\data\sample_messages.csv"
)

# New clean dataset
OUTPUT = Path(
    r"D:\Ai based OTP scam detection\dataset\otp_scam_seed_dataset.csv"
)

# Load dataset
df = pd.read_csv(SOURCE)

# Select only OTP Scam and Safe
df = df[df["label"].isin(["OTP Scam", "Safe"])].copy()

# Remove the 3 ambiguous OTP Scam examples
# OTP 01, OTP 30, OTP 31
otp_scam_exclude = {
    "Your OTP is [REDACTED_NUMBER]. Do not share it with anyone. Valid for [REDACTED_NUMBER] mins.",
    "Ola support: Share the driver OTP [REDACTED_NUMBER] with our team to investigate your complaint.",
    "Myntra: To process your return pickup, share OTP [REDACTED_NUMBER] with our logistics team.",
}

df = df[~df["text"].isin(otp_scam_exclude)]

# Remove the difficult Safe example
safe_exclude = {
    "Your package is out for delivery. OTP: [REDACTED_NUMBER] (share only with delivery agent)."
}

df = df[~df["text"].isin(safe_exclude)]

# Convert labels
df["label"] = df["label"].map({
    "OTP Scam": "SCAM",
    "Safe": "GENUINE"
})

# Keep only required columns
df = df[["text", "label"]]

# Remove exact duplicates
df = df.drop_duplicates(subset=["text"])

# Validation
print("\n===== DATASET VALIDATION =====")
print("Total rows:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

print("\nMissing text:", df["text"].isna().sum())
print("Missing labels:", df["label"].isna().sum())
print("Duplicate texts:", df["text"].duplicated().sum())

# Save
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nDataset created successfully:")
print(OUTPUT)