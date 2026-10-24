import pandas as pd
import os


# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = r"D:\Ai based OTP scam detection\dataset\SMSSpamCollection"


print("=" * 70)
print("SMS SPAM DATASET INSPECTION")
print("=" * 70)


# ============================================================
# 1. CHECK FILE
# ============================================================

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset file not found!")
    print(DATASET_PATH)
    exit()

print("\nDataset found successfully!")
print(DATASET_PATH)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC INFORMATION")
print("=" * 70)

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("2. CLASS DISTRIBUTION")
print("=" * 70)

print(df["label"].value_counts())

print("\nClass percentages:")
print(
    (df["label"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# ============================================================
# 6. EMPTY MESSAGES
# ============================================================

print("\n" + "=" * 70)
print("4. EMPTY MESSAGES")
print("=" * 70)

empty_messages = df["text"].astype(str).str.strip().eq("").sum()

print("Empty messages:", empty_messages)


# ============================================================
# 7. DUPLICATE MESSAGES
# ============================================================

print("\n" + "=" * 70)
print("5. DUPLICATE MESSAGES")
print("=" * 70)

duplicate_count = df["text"].duplicated().sum()

print("Duplicate messages:", duplicate_count)


# ============================================================
# 8. DUPLICATES WITH SAME/DIFFERENT LABEL
# ============================================================

print("\n" + "=" * 70)
print("6. CONFLICTING DUPLICATES")
print("=" * 70)

grouped = df.groupby("text")["label"].nunique()

conflicting_duplicates = (grouped > 1).sum()

print(
    "Messages appearing with different labels:",
    conflicting_duplicates
)


# ============================================================
# 9. MESSAGE LENGTH
# ============================================================

print("\n" + "=" * 70)
print("7. MESSAGE LENGTH")
print("=" * 70)

df["message_length"] = df["text"].astype(str).str.len()

print("Average length:", round(df["message_length"].mean(), 2))
print("Shortest message:", df["message_length"].min())
print("Longest message:", df["message_length"].max())


# ============================================================
# 10. SAMPLE HAM MESSAGES
# ============================================================

print("\n" + "=" * 70)
print("8. SAMPLE GENUINE (HAM) MESSAGES")
print("=" * 70)

ham_samples = df[df["label"] == "ham"]["text"].head(5)

for i, message in enumerate(ham_samples, 1):
    print(f"\nHAM {i}:")
    print(message)


# ============================================================
# 11. SAMPLE SPAM MESSAGES
# ============================================================

print("\n" + "=" * 70)
print("9. SAMPLE SPAM MESSAGES")
print("=" * 70)

spam_samples = df[df["label"] == "spam"]["text"].head(5)

for i, message in enumerate(spam_samples, 1):
    print(f"\nSPAM {i}:")
    print(message)


# ============================================================
# 12. CHECK LABELS
# ============================================================

print("\n" + "=" * 70)
print("10. UNIQUE LABELS")
print("=" * 70)

print(df["label"].unique())


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)

print("\nTotal messages:", len(df))
print("HAM messages:", (df["label"] == "ham").sum())
print("SPAM messages:", (df["label"] == "spam").sum())
print("Duplicates:", duplicate_count)
print("Empty messages:", empty_messages)
print("Conflicting duplicates:", conflicting_duplicates)

print("\nNo changes were made to the dataset.")
print("=" * 70)