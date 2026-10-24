"""
frontend.py
===========
Interactive Streamlit Web Application for 3-Class SMS Scam Detection.
Features:
  - 3-Class ML Prediction (SCAM, GENUINE, NON-SCAM SPAM)
  - Confidence percentage (calibrated)
  - Concrete Threat Indicators Breakdown
  - Transparent Risk Meter (0 to 100)
  - Actionable Cybersecurity Safety Guidance
  - Quick-Test Sample Presets
  - Robust Input Validation & Error Handling
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Allow importing scamguard
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scamguard.predictor import ScamGuard, ScamGuardOutput
from scamguard.config import CLASSES, CLASS_COLORS, CLASS_DESCRIPTIONS

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMS Scam Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING (CSS)
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 24px;
    }
    .result-card {
        padding: 24px;
        border-radius: 12px;
        margin-top: 16px;
        margin-bottom: 20px;
    }
    .card-scam {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
    }
    .card-spam {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
    }
    .card-genuine {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .badge-indicator {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL INSTANCE
# ============================================================

@st.cache_resource
def get_classifier():
    guard = ScamGuard()
    guard.ensure_ready()
    return guard

# Initialize session stats
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🛡️ System Guide")
    st.markdown(
        """
        This system classifies incoming SMS text messages into **three distinct categories**
        using Natural Language Processing and Machine Learning:
        
        * 🚨 **SCAM**
          Phishing, credential theft, fake bank alerts, lottery fraud, suspicious URLs.
        * 📢 **NON-SCAM SPAM**
          Marketing promotions, commercial advertisements, unwanted offers without clear fraud.
        * ✅ **GENUINE**
          Legitimate personal chats, service updates, and official transactional notifications.
        """
    )
    st.divider()

    st.subheader("Model Information")
    st.markdown(
        """
        * **Features:** TF-IDF (Unigrams & Bigrams)
        * **Model:** Balanced Linear Classification
        * **Evaluation Baseline:** 90.9% Logistic Regression / 87.9% Linear SVM
        * **Explainability:** Rule-Based Threat Analysis Layer
        """
    )
    st.divider()

    if st.session_state.history:
        st.subheader("Session Statistics")
        hist_df = pd.DataFrame(st.session_state.history)
        st.write(hist_df["class"].value_counts())
        if st.button("Clear History"):
            st.session_state.history.clear()
            st.rerun()

# ============================================================
# MAIN APPLICATION
# ============================================================

st.markdown('<div class="main-title">🛡️ SMS Scam Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time NLP classification, threat indicator detection, and risk analysis for mobile text messages</div>', unsafe_allow_html=True)

# Preset sample test messages
st.write("##### Quick Test Examples (Click to load):")
c1, c2, c3, c4, c5 = st.columns(5)

sample_text = ""
if c1.button("🚨 Fake Prize"):
    st.session_state.preset = "URGENT! You have won a £1,000 cash prize in our weekly draw! Call 09061701939 to claim code S89. Valid 12hrs only."
if c2.button("🔑 OTP Theft"):
    st.session_state.preset = "Dear Customer, your bank access is locked. Please share the 6-digit OTP code sent to your phone to restore your account immediately."
if c3.button("🏦 Fake Bank KYC"):
    st.session_state.preset = "HDFC Bank Alert: Your NetBanking account will be suspended today. Click http://hdfc-kyc-update.xyz to verify your PAN number."
if c4.button("📢 Promo Ad"):
    st.session_state.preset = "Great Summer Sale! Get 50% discount on all apparel at Trends this weekend only. Reply STOP to cancel."
if c5.button("✅ Personal Chat"):
    st.session_state.preset = "Hey, are you free for lunch today at 1pm? Let me know when you reach the cafeteria."

default_input = st.session_state.get("preset", "")

message_input = st.text_area(
    "Paste SMS message text to analyze:",
    value=default_input,
    height=140,
    placeholder="Type or paste an SMS message here..."
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🔍 Analyze SMS", type="primary", use_container_width=True)

# ============================================================
# PREDICTION & RESULTS
# ============================================================

if analyze_btn:
    if not message_input.strip():
        st.warning("⚠️ Please enter an SMS message before clicking Analyze.")
    else:
        try:
            with st.spinner("Analyzing message with NLP model..."):
                classifier = get_classifier()
                output: ScamGuardOutput = classifier.analyze(message_input)

            # Record history
            st.session_state.history.append({
                "text": message_input[:40] + "...",
                "class": output.predicted_class,
                "risk": output.risk_score
            })

            # Determine card style
            if output.predicted_class == "SCAM":
                card_class = "card-scam"
                icon = "🚨"
            elif output.predicted_class == "NON-SCAM SPAM":
                card_class = "card-spam"
                icon = "📢"
            else:
                card_class = "card-genuine"
                icon = "✅"

            st.divider()

            # Main Result Banner
            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <h2 style="margin: 0; color: {output.class_color};">{icon} Classified as: {output.predicted_class}</h2>
                    <p style="margin-top: 8px; font-size: 15px; color: #334155;">{output.class_description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Metric Columns
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Category", output.predicted_class)
            m2.metric("Model Confidence", f"{output.confidence_pct}%")
            m3.metric("Scam Risk Score", f"{output.risk_score} / 100")
            m4.metric("Threat Level", output.risk_level)

            st.divider()

            # Threat Indicators & Explainability Section
            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader("🔍 Analysis & Threat Indicators")
                st.write(f"**Explanation:** {output.explanation}")

                if output.detected_indicators:
                    st.write("**Specific Threat Triggers Detected:**")
                    for ind in output.detected_indicators:
                        st.markdown(f'<span class="badge badge-indicator">⚠️ {ind}</span>', unsafe_allow_html=True)
                else:
                    st.success("No explicit high-risk scam keywords or threat patterns found.")

                st.write("**Message Attributes:**")
                attr_badges = []
                if output.has_url:
                    attr_badges.append("🌐 Contains Link/URL")
                if output.has_phone:
                    attr_badges.append("📞 Contains Phone Number")
                if output.has_currency:
                    attr_badges.append("💰 Mentions Money/Currency")
                if not attr_badges:
                    attr_badges.append("📝 Plain text message")

                for b in attr_badges:
                    st.markdown(f'<span class="badge badge-info">{b}</span>', unsafe_allow_html=True)

            with right_col:
                st.subheader("💡 Safety Recommendations")
                for tip in output.tips:
                    st.write(tip)

                st.caption("Disclaimer: This tool assists in cybersecurity awareness and does not replace human judgment. Never share confidential credentials.")

        except Exception as err:
            st.error(f"Error during analysis: {err}")

# Footer
st.divider()
st.caption("SMS Scam Detection Using Natural Language Processing and Machine Learning | BCA Final Project")
