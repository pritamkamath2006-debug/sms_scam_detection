import pandas as pd

# Cleaned dataset
INPUT_PATH = r"D:\Ai based OTP scam detection\dataset\sms_scam_cleaned.csv"

print("=" * 70)
print("SPAM MESSAGE ANALYSIS")
print("=" * 70)

# Load dataset
df = pd.read_csv(INPUT_PATH, encoding="utf-8")

# Select spam messages
spam_df = df[df["label"] == "spam"].copy()

print(f"\nTotal spam messages: {len(spam_df)}")

# Basic statistics
spam_df["length"] = spam_df["text"].str.len()

print("\nMessage length statistics:")
print(spam_df["length"].describe())

# Show first 30 spam messages
print("\n" + "=" * 70)
print("SAMPLE SPAM MESSAGES")
print("=" * 70)

for i, message in enumerate(spam_df["text"].head(30), start=1):
    print(f"\n{i}. {message}")

# Common words
print("\n" + "=" * 70)
print("COMMON WORDS IN SPAM MESSAGES")
print("=" * 70)

text = " ".join(spam_df["text"].astype(str)).lower()

# Simple word frequency
words = text.split()

from collections import Counter

word_counts = Counter(words)

for word, count in word_counts.most_common(30):
    print(f"{word}: {count}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)