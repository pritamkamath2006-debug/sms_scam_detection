from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# SETTINGS
# ============================================================

INPUT_PATH = BASE_DIR / "dataset" / "sms_scam_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "sms_scam_categorized.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("SMS SCAM CATEGORY ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_PATH, encoding="utf-8")

print(f"\nTotal messages loaded: {len(df)}")


# ============================================================
# SCAM KEYWORDS / PATTERNS
# ============================================================

CATEGORY_PATTERNS = {

    "Prize/Lottery Scam": [
        r"\bwon\b",
        r"\bwinner\b",
        r"\bwinning\b",
        r"\bprize\b",
        r"\breward\b",
        r"\blottery\b",
        r"\bjackpot\b",
        r"\bclaim\b",
        r"\bcash prize\b",
        r"\bbonus\b",
    ],

    "Financial Scam": [
        r"\bcash\b",
        r"\bloan\b",
        r"\bbonus\b",
        r"\baccount\b",
        r"\bpayment\b",
        r"\brefund\b",
        r"\bcredit\b",
        r"\bdebit\b",
        r"\bbank\b",
        r"\bmoney\b",
        r"\b£\d+",
        r"\$\d+",
    ],

    "Phishing/Link Scam": [
        r"http[s]?://",
        r"www\.",
        r"\bclick\b",
        r"\blink\b",
        r"\bverify\b",
        r"\bverification\b",
        r"\bsecure\b",
        r"\blogin\b",
        r"\bupdate your account\b",
    ],

    "Premium-Call Scam": [
        r"\bcall\b",
        r"\bcall now\b",
        r"\bcall to claim\b",
        r"\bpremium\b",
        r"\btext now\b",
        r"\breply\b",
        r"\btxt\b",
    ],

    "Subscription Scam": [
        r"\bsubscription\b",
        r"\bsubscribe\b",
        r"\bmonthly\b",
        r"\bper month\b",
        r"\bservice\b",
        r"\bstop\b",
        r"\bcharged\b",
    ],

    "Job Scam": [
        r"\bjob\b",
        r"\bjobs\b",
        r"\bwork from home\b",
        r"\bcareer\b",
        r"\bsalary\b",
        r"\bemployment\b",
        r"\bhiring\b",
        r"\brecruitment\b",
    ],

    "Delivery/Support Scam": [
        r"\bdelivery\b",
        r"\bdeliver\b",
        r"\bparcel\b",
        r"\bpackage\b",
        r"\bcustomer service\b",
        r"\bcustomer care\b",
        r"\bsupport\b",
        r"\border\b",
    ],

    "OTP Scam": [
        r"\botp\b",
        r"\bone time password\b",
        r"\bverification code\b",
        r"\bsecurity code\b",
        r"\bpasscode\b",
    ]
}


# ============================================================
# CATEGORY DETECTION FUNCTION
# ============================================================

def detect_categories(message):

    message = str(message).lower()

    detected_categories = []

    for category, patterns in CATEGORY_PATTERNS.items():

        for pattern in patterns:

            if re.search(pattern, message):

                detected_categories.append(category)

                break

    return detected_categories


# ============================================================
# ANALYZE SPAM MESSAGES
# ============================================================

print("\nAnalyzing spam messages...")

df["detected_categories"] = ""

df["scam_indicator_count"] = 0

df["likely_scam"] = "NO"


for index, row in df.iterrows():

    message = str(row["text"])
    label = str(row["label"]).lower()

    # Only analyze spam messages
    if label == "spam":

        categories = detect_categories(message)

        df.at[index, "detected_categories"] = ", ".join(categories)

        df.at[index, "scam_indicator_count"] = len(categories)

        # At least one suspicious category
        if len(categories) > 0:
            df.at[index, "likely_scam"] = "YES"


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SCAM CATEGORY RESULTS")
print("=" * 70)

spam_df = df[df["label"] == "spam"].copy()

print(f"\nTotal spam messages: {len(spam_df)}")

likely_scam_count = len(
    spam_df[spam_df["likely_scam"] == "YES"]
)

print(f"Likely scam messages: {likely_scam_count}")

print(
    f"Spam messages without detected scam indicators: "
    f"{len(spam_df) - likely_scam_count}"
)


# ============================================================
# CATEGORY COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY COUNTS")
print("=" * 70)

category_counts = {}

for category in CATEGORY_PATTERNS.keys():

    count = 0

    for categories in spam_df["detected_categories"]:

        if category in str(categories).split(", "):

            count += 1

    category_counts[category] = count


for category, count in category_counts.items():

    print(f"{category}: {count}")


# ============================================================
# SHOW LIKELY SCAM EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE LIKELY SCAM MESSAGES")
print("=" * 70)

scam_examples = spam_df[
    spam_df["likely_scam"] == "YES"
].head(30)


for i, (_, row) in enumerate(
    scam_examples.iterrows(),
    start=1
):

    print(f"\n{i}. {row['text']}")

    print(
        f"   Category: "
        f"{row['detected_categories']}"
    )


# ============================================================
# SAVE RESULT
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nCategorized dataset saved to:")

print(OUTPUT_PATH)

print("\nOriginal dataset was NOT modified.")