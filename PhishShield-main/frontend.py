import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_FILE = PROJECT_DIR / "dataset" / "sms_scam_cleaned.csv"

ARTIFACT_DIR = BASE_DIR / "scamguard" / "artifacts"

MODEL_FILE = ARTIFACT_DIR / "best_model.joblib"
VECTORIZER_FILE = ARTIFACT_DIR / "tfidf_vectorizer.joblib"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMS Scam Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
    }

    .scam-box {
        background-color: #ffe5e5;
        border: 2px solid #e53935;
    }

    .genuine-box {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
    }

    .big-result {
        font-size: 30px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not DATASET_FILE.exists():
        return None

    try:
        return pd.read_csv(DATASET_FILE)
    except Exception:
        return None


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():
        return None, None

    if not VECTORIZER_FILE.exists():
        return None, None

    try:

        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)

        return model, vectorizer

    except Exception:

        return None, None


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Replace URLs
    text = re.sub(
        r"http\S+|www\S+",
        " URL ",
        text
    )

    # Replace email addresses
    text = re.sub(
        r"\S+@\S+",
        " EMAIL ",
        text
    )

    # Replace numbers
    text = re.sub(
        r"\d+",
        " NUMBER ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# SUSPICIOUS PATTERN DETECTION
# ============================================================

def detect_suspicious_patterns(text):

    text_lower = text.lower()

    patterns = []

    # URL
    if re.search(
        r"http[s]?://|www\.",
        text_lower
    ):
        patterns.append(
            "Contains a web link"
        )

    # OTP / sensitive information
    if re.search(
        r"\botp\b|verification code|verify|pin|password",
        text_lower
    ):
        patterns.append(
            "Mentions sensitive verification information"
        )

    # Prize / reward
    if re.search(
        r"\bprize\b|\bwon\b|\bwinner\b|\blottery\b|\breward\b|cash prize",
        text_lower
    ):
        patterns.append(
            "Mentions a prize or reward"
        )

    # Urgency
    if re.search(
        r"\burgent\b|immediately|act now|expires|last chance",
        text_lower
    ):
        patterns.append(
            "Uses urgency or pressure"
        )

    # Money
    if re.search(
        r"\bpayment\b|\bpay\b|\bmoney\b|\bfee\b|\brefund\b|\baccount\b",
        text_lower
    ):
        patterns.append(
            "Mentions money or payment"
        )

    # Action request
    if re.search(
        r"\bclick\b|\btap\b|call now|\breply\b|\bcontact\b",
        text_lower
    ):
        patterns.append(
            "Contains an action request"
        )

    # Account problem
    if re.search(
        r"\bblocked\b|\bsuspended\b|\bdeactivated\b|\bclosed\b",
        text_lower
    ):
        patterns.append(
            "Claims an account or service problem"
        )

    # Phone number
    if re.search(
        r"(?:\+?\d[\d\s\-]{7,}\d)",
        text
    ):
        patterns.append(
            "Contains a phone number"
        )

    return patterns


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(text, prediction):

    patterns = detect_suspicious_patterns(text)

    score = 0

    # Model prediction contribution
    if prediction == 1:
        score += 30

    # Suspicious pattern contribution
    score += min(
        len(patterns) * 10,
        50
    )

    # URL
    if re.search(
        r"http[s]?://|www\.",
        text.lower()
    ):
        score += 10

    # Sensitive information
    if re.search(
        r"\botp\b|password|pin|verification code",
        text.lower()
    ):
        score += 10

    return min(
        score,
        100
    )


# ============================================================
# SAFETY TIPS
# ============================================================

def get_safety_tips(prediction):

    if prediction == 1:

        return [
            "Do not click suspicious links.",
            "Do not share OTPs, passwords, PINs or banking details.",
            "Verify the sender through an official source.",
            "Do not make payments based only on the message.",
            "Block or report the sender if the message is suspicious."
        ]

    return [
        "Still verify unexpected requests independently.",
        "Never share OTPs, passwords or PINs with anyone.",
        "Check links carefully before opening them."
    ]


# ============================================================
# SMS PREDICTION
# ============================================================

def predict_sms(message, model, vectorizer):

    cleaned_message = clean_text(message)

    # Convert text to TF-IDF
    features = vectorizer.transform(
        [cleaned_message]
    )

    # ML prediction
    prediction = int(
        model.predict(features)[0]
    )

    # Linear SVM decision score
    decision_score = float(
        model.decision_function(features)[0]
    )

    # Confidence-like score
    #
    # IMPORTANT:
    # This is NOT a calibrated probability.
    # It is derived from the SVM decision function.
    confidence = min(
        99,
        max(
            50,
            int(
                50 + abs(decision_score) * 10
            )
        )
    )

    # Current model:
    # 0 = HAM
    # 1 = SPAM

    if prediction == 1:

        label = "SCAM / SPAM"

    else:

        label = "GENUINE / HAM"

    # Suspicious patterns
    patterns = detect_suspicious_patterns(
        message
    )

    # Risk score
    risk_score = calculate_risk_score(
        message,
        prediction
    )

    # Explanation
    if prediction == 1:

        if patterns:

            reason = (
                "The machine-learning model classified "
                "this message as suspicious. "
                "The message also contains: "
                + ", ".join(patterns)
                + "."
            )

        else:

            reason = (
                "The machine-learning model classified "
                "this message as suspicious."
            )

    else:

        if patterns:

            reason = (
                "The machine-learning model classified "
                "this message as HAM, although some "
                "potentially suspicious patterns were detected: "
                + ", ".join(patterns)
                + "."
            )

        else:

            reason = (
                "The machine-learning model classified "
                "this message as HAM and no strong "
                "suspicious patterns were detected."
            )

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "risk_score": risk_score,
        "patterns": patterns,
        "reason": reason
    }


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = load_dataset()

model, vectorizer = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ SMS Scam Detector"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Analyze SMS",
        "📊 Dataset",
        "🤖 Model",
        "ℹ️ About"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '🛡️ SMS Scam Detection Using NLP'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning based SMS classification system'
        '</div>',
        unsafe_allow_html=True
    )

    # Dataset statistics
    if df is not None:

        total_messages = len(df)

        if "label" in df.columns:

            ham_count = len(
                df[
                    df["label"]
                    .astype(str)
                    .str.upper()
                    .eq("HAM")
                ]
            )

            spam_count = len(
                df[
                    df["label"]
                    .astype(str)
                    .str.upper()
                    .eq("SPAM")
                ]
            )

        else:

            ham_count = 0
            spam_count = 0

    else:

        total_messages = 0
        ham_count = 0
        spam_count = 0

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Cleaned Messages",
            total_messages
        )

    with col2:

        st.metric(
            "HAM Messages",
            ham_count
        )

    with col3:

        st.metric(
            "SPAM Messages",
            spam_count
        )

    st.divider()

    # Pipeline
    st.subheader(
        "🔄 Project Pipeline"
    )

    st.markdown(
        """
        **SMS Message**

        ↓

        **Text Preprocessing**

        ↓

        **TF-IDF Feature Extraction**

        ↓

        **Linear SVM**

        ↓

        **Prediction**

        ↓

        **Risk Analysis + Explanation**

        ↓

        **Safety Recommendation**
        """
    )

    st.divider()

    st.info(
        "The current trained model is a HAM/SPAM "
        "classification baseline. Its metrics should "
        "not be presented as final SCAM-vs-GENUINE "
        "ground-truth performance."
    )


# ============================================================
# ANALYZE SMS
# ============================================================

elif page == "🔍 Analyze SMS":

    st.title(
        "🔍 Analyze an SMS"
    )

    st.write(
        "Enter an SMS message below and the trained "
        "machine-learning model will classify it."
    )

    message = st.text_area(
        "Enter SMS message",
        height=180,
        placeholder=(
            "Example: Congratulations! "
            "You have won a prize. "
            "Click the link to claim..."
        )
    )

    analyze_button = st.button(
        "🔎 Analyze SMS",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not message.strip():

            st.warning(
                "Please enter an SMS message."
            )

        elif model is None or vectorizer is None:

            st.error(
                "Trained model files were not found."
            )

            st.code(
                "python scripts\\train_models.py"
            )

        else:

            result = predict_sms(
                message,
                model,
                vectorizer
            )

            st.divider()

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if result["prediction"] == 1:

                st.markdown(
                    f"""
                    <div class="result-box scam-box">
                        <div class="big-result">
                            🚨 {result["label"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-box genuine-box">
                        <div class="big-result">
                            ✅ {result["label"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # Metrics
            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Model Confidence",
                    f"{result['confidence']}%"
                )

            with col2:

                st.metric(
                    "Risk Score",
                    f"{result['risk_score']}/100"
                )

            # Explanation
            st.subheader(
                "🧠 Why was it classified this way?"
            )

            st.write(
                result["reason"]
            )

            # Patterns
            if result["patterns"]:

                st.subheader(
                    "⚠️ Detected Patterns"
                )

                for pattern in result["patterns"]:

                    st.write(
                        f"• {pattern}"
                    )

            else:

                st.success(
                    "No strong suspicious patterns were detected."
                )

            # Safety
            st.subheader(
                "🛡️ Safety Recommendations"
            )

            tips = get_safety_tips(
                result["prediction"]
            )

            for tip in tips:

                st.write(
                    f"• {tip}"
                )

            # Message
            st.subheader(
                "📩 Message Analyzed"
            )

            st.code(
                message,
                language=None
            )

            st.caption(
                "Note: The displayed confidence is a "
                "confidence-like score derived from the "
                "Linear SVM decision function. It is not "
                "a calibrated probability."
            )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.title(
        "📊 Dataset"
    )

    if df is None:

        st.error(
            "Dataset file was not found."
        )

    else:

        st.write(
            f"Total cleaned records: **{len(df)}**"
        )

        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "📈 Class Distribution"
        )

        if "label" in df.columns:

            counts = (
                df["label"]
                .value_counts()
            )

            st.bar_chart(
                counts
            )

        st.warning(
            "The current dataset uses HAM/SPAM labels. "
            "HAM/SPAM should not automatically be interpreted "
            "as SCAM/GENUINE."
        )

        st.markdown(
            """
            **Current cleaned dataset:**

            - Total: **5,158**
            - HAM: **4,516**
            - SPAM: **642**
            """
        )


# ============================================================
# MODEL PAGE
# ============================================================

elif page == "🤖 Model":

    st.title(
        "🤖 Machine Learning Model"
    )

    st.write(
        "Four machine-learning models were trained and "
        "evaluated using the current HAM/SPAM SMS dataset."
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "🔄 Machine Learning Pipeline"
    )

    st.code(
        """
SMS Message
     ↓
Text Preprocessing
     ↓
TF-IDF Feature Extraction
     ↓
Machine Learning Model
     ↓
Prediction
     ↓
Risk Analysis + Explanation
        """
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "📊 Model Comparison"
    )

    results_data = {
        "Model": [
            "Naive Bayes",
            "Logistic Regression",
            "Linear SVM",
            "Random Forest"
        ],

        "Accuracy": [
            95.54,
            97.77,
            98.35,
            97.00
        ],

        "Precision": [
            100.00,
            88.32,
            93.70,
            100.00
        ],

        "Recall": [
            64.06,
            94.53,
            92.97,
            75.78
        ],

        "F1 Score": [
            78.10,
            91.32,
            93.33,
            86.22
        ]
    }

    results_df = pd.DataFrame(
        results_data
    )

    display_df = results_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_df[column] = (
            display_df[column]
            .map(
                lambda x: f"{x:.2f}%"
            )
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # PERFORMANCE CHART
    # --------------------------------------------------------

    st.subheader(
        "📈 Model Performance"
    )

    chart_df = results_df.set_index(
        "Model"
    )

    st.bar_chart(
        chart_df[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ]
    )

    # --------------------------------------------------------
    # SELECTED MODEL
    # --------------------------------------------------------

    st.subheader(
        "🏆 Selected Model"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "Linear SVM"
        )

    with col2:

        st.metric(
            "Accuracy",
            "98.35%"
        )

    with col3:

        st.metric(
            "F1 Score",
            "93.33%"
        )

    st.success(
        "Linear SVM was selected as the primary model "
        "because it achieved the best F1-score among "
        "the four evaluated models on the current "
        "HAM/SPAM baseline."
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "🔢 Linear SVM Confusion Matrix"
    )

    st.write(
        "Results from the held-out test set."
    )

    confusion_matrix = pd.DataFrame(
        [
            [896, 8],
            [9, 119]
        ],
        index=[
            "Actual HAM",
            "Actual SPAM"
        ],
        columns=[
            "Predicted HAM",
            "Predicted SPAM"
        ]
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        """
        **Interpretation:**

        - **896** → HAM correctly classified as HAM
        - **8** → HAM incorrectly classified as SPAM
        - **9** → SPAM incorrectly classified as HAM
        - **119** → SPAM correctly classified as SPAM
        """
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "📁 Training Dataset"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Messages",
            "5,158"
        )

    with col2:

        st.metric(
            "HAM",
            "4,516"
        )

    with col3:

        st.metric(
            "SPAM",
            "642"
        )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    st.subheader(
        "🔤 TF-IDF Feature Extraction"
    )

    st.write(
        "TF-IDF converts SMS text into numerical features "
        "that can be processed by the machine-learning model."
    )

    st.markdown(
        """
        **Configuration:**

        - N-gram range: **1–2**
        - Maximum features: **15,000**
        - Sublinear TF: **Enabled**
        - Features: **Unigrams + Bigrams**
        """
    )

    # --------------------------------------------------------
    # SVM CONFIGURATION
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Linear SVM Configuration"
    )

    st.code(
        """
TF-IDF
ngram_range = (1, 2)
max_features = 15000
sublinear_tf = True

Linear SVM
Training/Test Split = 80/20
Random State = 42
        """
    )

    # --------------------------------------------------------
    # RESEARCH NOTE
    # --------------------------------------------------------

    st.warning(
        "Research note: These metrics represent the current "
        "HAM/SPAM baseline dataset. They should not be "
        "presented as final SCAM-vs-GENUINE ground-truth "
        "performance."
    )

    st.info(
        "The next research step is to validate scam-specific "
        "labels using manually reviewed SMS messages before "
        "reporting final scam-detection performance."
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ## SMS Scam Detection Using NLP and Machine Learning

        This project analyzes SMS messages using Natural Language
        Processing and Machine Learning techniques.

        ### Technologies Used

        - Python
        - Streamlit
        - Scikit-learn
        - TF-IDF
        - Linear SVM
        - Pandas
        - Joblib

        ### Main Objective

        The system analyzes an SMS message and provides:

        - Machine-learning classification
        - Confidence-like score
        - Risk score
        - Suspicious pattern detection
        - Explanation
        - Safety recommendations

        ### Machine Learning Pipeline

        SMS → Text Preprocessing → TF-IDF → Linear SVM → Prediction

        ### Important Research Note

        The current trained baseline uses HAM/SPAM labels.

        Therefore, the current model metrics should be described
        as a HAM/SPAM classification baseline and not as final
        SCAM-vs-GENUINE ground-truth accuracy.

        A manually validated scam-specific dataset is required
        before claiming final scam-detection performance.
        """
    )