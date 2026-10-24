import pandas as pd
import os

# ============================================================
# SMS DATASET CLEANING
# ============================================================

# Original dataset
INPUT_PATH = r"D:\Ai based OTP scam detection\dataset\SMSSpamCollection"

# New cleaned dataset
OUTPUT_PATH = r"D:\Ai based OTP scam detection\dataset\sms_scam_cleaned.csv"


print("=" * 70)
print("SMS DATASET CLEANING")
print("=" * 70)


# ============================================================
# 1. CHECK FILE
# ============================================================

if not os.path.exists(INPUT_PATH):
    print("ERROR: Dataset file not found!")
    print(INPUT_PATH)
    exit()

print("\nDataset found successfully.")


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

print(f"\nOriginal rows: {len(df)}")


# ============================================================
# 3. REMOVE MISSING / EMPTY MESSAGES
# ============================================================

df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()

before_empty = len(df)

df = df[
    (df["text"] != "") &
    (df["text"].notna())
].copy()

empty_removed = before_empty - len(df)

print(f"Empty messages removed: {empty_removed}")


# ============================================================
# 4. CHECK CONFLICTING DUPLICATES
# ============================================================

conflicting = (
    df.groupby("text")["label"]
    .nunique()
)

conflicting_messages = conflicting[conflicting > 1]

print(f"Conflicting duplicate messages: {len(conflicting_messages)}")


# ============================================================
# 5. REMOVE DUPLICATE MESSAGES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["text"],
    keep="first"
).reset_index(drop=True)

duplicates_removed = before_duplicates - len(df)

print(f"Duplicate messages removed: {duplicates_removed}")


# ============================================================
# 6. CHECK FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(f"Total messages: {len(df)}")

print("\nClass distribution:")

print(df["label"].value_counts())


# ============================================================
# 7. CHECK FOR REMAINING DUPLICATES
# ============================================================

remaining_duplicates = df["text"].duplicated().sum()

print(f"\nRemaining duplicate messages: {remaining_duplicates}")


# ============================================================
# 8. CHECK FOR MISSING VALUES
# ============================================================

print("\nMissing values:")

print(df.isnull().sum())


# ============================================================
# 9. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print(f"\nCleaned dataset saved to:")
print(OUTPUT_PATH)

print("\nOriginal dataset was NOT modified.")