"""
test_sms_scam_detection.py
==========================
Comprehensive automated test suite for the SMS Scam Detection System.
Tests cover:
  1. Text Preprocessing & Tokenization
  2. Dataset Integrity & 3-Class Validation
  3. Model Loading & Feature Vectorizer
  4. 3-Class Prediction Output & Edge Cases
  5. Real-World Scenario Evaluations (Personal, Transactional, Promo, Fake Prize, OTP Theft, Bank Alert)
  6. Threat Indicators & Transparent Explainability Layer
"""

import sys
import os
import unittest
from pathlib import Path
import pandas as pd
import joblib

# Paths setup
REPO_ROOT = Path(__file__).resolve().parents[1]
PHISH_ROOT = REPO_ROOT / "PhishShield-main"

if str(PHISH_ROOT) not in sys.path:
    sys.path.insert(0, str(PHISH_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scamguard.preprocessing import clean_text
from scamguard.explain import analyze_risk_indicators, compute_composite_risk
from scamguard.predictor import ScamGuard
from scamguard.config import CLASSES, SCAM_INDICATORS


class TestTextPreprocessing(unittest.TestCase):
    """Verifies that text preprocessing is clean, consistent, and does not lose indicators."""

    def test_empty_and_none_text(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   \n\t  "), "")

    def test_lowercasing_and_whitespace(self):
        raw = "  URGENT!   PLEASE CALL NOW \n\n 0800  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "urgent! please call now 0800")

    def test_preserves_indicators(self):
        raw = "Win £1000 cash! Visit https://claim.xyz or call 9876543210. OTP is 1234."
        cleaned = clean_text(raw)
        self.assertIn("£1000", cleaned)
        self.assertIn("https://claim.xyz", cleaned)
        self.assertIn("9876543210", cleaned)
        self.assertIn("otp", cleaned)


class TestDatasetIntegrity(unittest.TestCase):
    """Verifies the validated 3-class dataset quality and absence of leaks/corruption."""

    def test_reviewed_dataset_exists_and_valid(self):
        path = REPO_ROOT / "dataset" / "reviewed_dataset.csv"
        self.assertTrue(path.exists(), "reviewed_dataset.csv must exist.")

        df = pd.read_csv(path)
        self.assertGreaterEqual(len(df), 146, "Reviewed dataset must have at least 146 verified rows.")
        self.assertEqual(df["text"].isna().sum(), 0, "No text should be missing.")
        self.assertEqual(df["text"].duplicated().sum(), 0, "No duplicate texts allowed.")

        labels = set(df["manual_label"].unique())
        for lab in labels:
            self.assertIn(lab, CLASSES, f"Label {lab} must be one of {CLASSES}")


class TestModelArtifacts(unittest.TestCase):
    """Verifies that model weights and TF-IDF vectorizer are properly serialized and loadable."""

    def setUp(self):
        self.guard = ScamGuard()
        self.guard.ensure_ready()

    def test_model_files_exist(self):
        self.assertTrue(self.guard.model_path.exists(), "best_model.joblib must exist.")
        self.assertTrue(self.guard.vectorizer_path.exists(), "tfidf_vectorizer.joblib must exist.")

    def test_vectorizer_properties(self):
        vec = joblib.load(self.guard.vectorizer_path)
        self.assertGreater(len(vec.vocabulary_), 500, "TF-IDF vocabulary must contain features.")


class TestPredictionOutput(unittest.TestCase):
    """Verifies prediction outputs, edge case handling, and confidence values."""

    def setUp(self):
        self.guard = ScamGuard()
        self.guard.ensure_ready()

    def test_empty_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.guard.analyze("")
        with self.assertRaises(ValueError):
            self.guard.analyze("    ")
        with self.assertRaises(ValueError):
            self.guard.analyze(None)

    def test_prediction_returns_valid_class(self):
        res = self.guard.analyze("Hello, are we still meeting today at the library?")
        self.assertIn(res.predicted_class, CLASSES)
        self.assertGreaterEqual(res.confidence_score, 0.0)
        self.assertLessEqual(res.confidence_score, 1.0)
        self.assertGreaterEqual(res.risk_score, 0)
        self.assertLessEqual(res.risk_score, 100)


class TestRealWorldScenarios(unittest.TestCase):
    """
    Evaluates real-world SMS examples across:
      - Genuine Personal
      - Genuine Transactional
      - Promotional Ad
      - Fake Prize Scam
      - OTP Theft Scam
      - Fake Bank Alert
    """

    def setUp(self):
        self.guard = ScamGuard()
        self.guard.ensure_ready()

    def test_genuine_personal(self):
        text = "Hey mom, I reached college safely. Will call you after class."
        res = self.guard.analyze(text)
        self.assertEqual(res.predicted_class, "GENUINE")
        self.assertLessEqual(res.risk_score, 35)

    def test_genuine_transactional(self):
        text = "Your package has been delivered to your front door. Thank you for shopping with us."
        res = self.guard.analyze(text)
        self.assertEqual(res.predicted_class, "GENUINE")
        self.assertLessEqual(res.risk_score, 35)

    def test_promotional_ad(self):
        text = "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"
        res = self.guard.analyze(text)
        self.assertIn(res.predicted_class, ["NON-SCAM SPAM", "GENUINE"])
        self.assertLessEqual(res.risk_score, 55)

    def test_fake_prize_scam(self):
        text = "URGENT! You have won a £1000 cash prize in our weekly lottery draw! Call 09061701939 to claim code S89."
        res = self.guard.analyze(text)
        self.assertEqual(res.predicted_class, "SCAM")
        self.assertGreaterEqual(res.risk_score, 65)
        self.assertTrue(any("prize" in ind.lower() or "urgent" in ind.lower() for ind in res.detected_indicators))

    def test_otp_theft_scam(self):
        text = "Dear Customer, your bank access is locked. Share the 6-digit verification code OTP immediately to verify."
        res = self.guard.analyze(text)
        self.assertEqual(res.predicted_class, "SCAM")
        self.assertGreaterEqual(res.risk_score, 70)
        self.assertTrue(any("otp" in ind.lower() for ind in res.detected_indicators))

    def test_fake_bank_alert(self):
        text = "HDFC Alert: Your account is suspended due to pending KYC. Click http://hdfc-kyc-verify.xyz to update."
        res = self.guard.analyze(text)
        self.assertEqual(res.predicted_class, "SCAM")
        self.assertGreaterEqual(res.risk_score, 70)
        self.assertTrue(res.has_url)


class TestRiskAnalysisLayer(unittest.TestCase):
    """Verifies that the rule-based threat indicator layer functions transparently."""

    def test_all_indicator_categories(self):
        samples = {
            "otp_password": "Share your OTP and verification code.",
            "account_urgency": "Your account will be suspended today urgently.",
            "payment_money": "Please transfer money and pay the processing fee now.",
            "fake_prize": "Congratulations you won the jackpot prize reward.",
            "suspicious_link": "Visit http://tinyurl.com/free-claim now.",
            "confidential_data": "Enter your card number and cvv details.",
        }

        for cat_name, phrase in samples.items():
            info = analyze_risk_indicators(phrase)
            self.assertGreater(len(info["detected_indicators"]), 0, f"Failed to detect {cat_name}")


if __name__ == "__main__":
    unittest.main()
