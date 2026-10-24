import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMS Manual Labeling",
    page_icon="🏷️",
    layout="wide"
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "dataset" / "sms_scam_validation.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "manual_labels.csv"

TARGET_PER_CLASS = 100

LABELS = [
    "SCAM",
    "GENUINE",
    "NON-SCAM SPAM"
]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 35px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .message-box {
        background-color: #f5f7fa;
        border: 1px solid #d9dee7;
        border-radius: 12px;
        padding: 25px;
        font-size: 22px;
        line-height: 1.6;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #eef5ff;
        border: 1px solid #c9ddff;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================

if "skipped_ids" not in st.session_state:
    st.session_state.skipped_ids = set()

if "current_message_id" not in st.session_state:
    st.session_state.current_message_id = None

# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

@st.cache_data
def load_validation_data():

    if not INPUT_FILE.exists():
        return None

    df = pd.read_csv(INPUT_FILE)

    # Check for text column
    if "text" not in df.columns:

        possible_columns = [
            "message",
            "sms",
            "text_transcript",
            "Message"
        ]

        found_column = None

        for column in possible_columns:
            if column in df.columns:
                found_column = column
                break

        if found_column is not None:
            df = df.rename(columns={found_column: "text"})
        else:
            return None

    # Create message ID if missing
    if "message_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["message_id"] = range(1, len(df) + 1)

    df["message_id"] = df["message_id"].astype(str)

    df["text"] = df["text"].fillna("").astype(str)

    # Remove empty messages
    df = df[df["text"].str.strip() != ""].copy()

    df.reset_index(drop=True, inplace=True)

    return df

# ============================================================
# LOAD PREVIOUS LABELS
# ============================================================

def load_previous_labels():

    if not OUTPUT_FILE.exists():

        return pd.DataFrame(
            columns=[
                "message_id",
                "text",
                "original_label",
                "manual_label"
            ]
        )

    try:

        labels_df = pd.read_csv(OUTPUT_FILE)

        required_columns = [
            "message_id",
            "text",
            "original_label",
            "manual_label"
        ]

        for column in required_columns:

            if column not in labels_df.columns:
                labels_df[column] = ""

        labels_df["message_id"] = (
            labels_df["message_id"].astype(str)
        )

        return labels_df[required_columns]

    except Exception as e:

        st.error(
            f"Error loading previous labels: {e}"
        )

        return pd.DataFrame(
            columns=[
                "message_id",
                "text",
                "original_label",
                "manual_label"
            ]
        )

# ============================================================
# SAVE LABEL
# ============================================================

def save_label(
    message_id,
    text,
    original_label,
    manual_label
):

    labels_df = load_previous_labels()

    # Remove previous label for this message
    labels_df = labels_df[
        labels_df["message_id"].astype(str)
        != str(message_id)
    ].copy()

    new_row = pd.DataFrame(
        [
            {
                "message_id": str(message_id),
                "text": text,
                "original_label": original_label,
                "manual_label": manual_label
            }
        ]
    )

    labels_df = pd.concat(
        [labels_df, new_row],
        ignore_index=True
    )

    labels_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

# ============================================================
# GET ORIGINAL LABEL
# ============================================================

def get_original_label(row):

    possible_columns = [
        "label",
        "original_label",
        "category",
        "spam_label",
        "class"
    ]

    for column in possible_columns:

        if column in row.index:

            value = row[column]

            if pd.notna(value):
                return str(value)

    return ""

# ============================================================
# GET LABEL COUNTS
# ============================================================

def get_label_counts(labels_df):

    if labels_df.empty:

        return {
            "SCAM": 0,
            "GENUINE": 0,
            "NON-SCAM SPAM": 0
        }

    counts = labels_df["manual_label"].value_counts()

    return {
        "SCAM": int(counts.get("SCAM", 0)),
        "GENUINE": int(counts.get("GENUINE", 0)),
        "NON-SCAM SPAM": int(
            counts.get("NON-SCAM SPAM", 0)
        )
    }

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🏷️ SMS Manual Labeling</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Manually classify SMS messages for the SMS Scam Detection project.'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# LOAD DATA
# ============================================================

df = load_validation_data()

if df is None:

    st.error(
        f"Dataset not found or required text column is missing.\n\n"
        f"Expected file:\n`{INPUT_FILE}`"
    )

    st.stop()

labels_df = load_previous_labels()

# ============================================================
# LABEL COUNTS
# ============================================================

counts = get_label_counts(labels_df)

total_labeled = sum(counts.values())

# ============================================================
# PROGRESS
# ============================================================

st.subheader("📊 Labeling Progress")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🚨 SCAM",
        counts["SCAM"],
        f"Target: {TARGET_PER_CLASS}"
    )

with col2:

    st.metric(
        "✅ GENUINE",
        counts["GENUINE"],
        f"Target: {TARGET_PER_CLASS}"
    )

with col3:

    st.metric(
        "📢 NON-SCAM SPAM",
        counts["NON-SCAM SPAM"],
        f"Target: {TARGET_PER_CLASS}"
    )

with col4:

    st.metric(
        "📝 TOTAL LABELED",
        total_labeled
    )

# ============================================================
# OVERALL PROGRESS
# ============================================================

total_target = TARGET_PER_CLASS * len(LABELS)

progress = min(
    total_labeled / total_target,
    1.0
)

st.progress(progress)

st.write(
    f"Overall progress: "
    f"**{total_labeled} / {total_target}** labels"
)

# ============================================================
# LABELING RULES
# ============================================================

with st.expander(
    "📖 Labeling Rules",
    expanded=False
):

    st.markdown(
        """
        ### 🚨 SCAM

        Choose **SCAM** when the message attempts to deceive,
        commit fraud, steal money, or steal personal information.

        Examples:

        - Fake prize or lottery
        - Fake bank/security warning
        - Suspicious payment request
        - Phishing link
        - Fake delivery/support message
        - OTP/password stealing
        - Fake job or financial opportunity

        ---

        ### ✅ GENUINE

        Choose **GENUINE** when the message appears legitimate
        and is not a scam or promotional spam.

        Examples:

        - Normal conversation
        - Genuine service notification
        - Appointment reminder
        - College/work communication

        ---

        ### 📢 NON-SCAM SPAM

        Choose **NON-SCAM SPAM** when the message is advertising
        or promotional spam but there is no clear evidence of fraud.

        Examples:

        - Commercial advertisements
        - Promotional offers
        - Paid subscriptions
        - Marketing messages
        - Promotional services

        ---

        ### 🔄 SKIP

        Choose **SKIP** if you are unsure.

        The message will temporarily be removed from the current
        review queue and can be reviewed later.
        """
    )

# ============================================================
# FIND LABELED IDS
# ============================================================

labeled_ids = set(
    labels_df["message_id"]
    .astype(str)
    .tolist()
)

# ============================================================
# FIND SKIPPED IDS
# ============================================================

skipped_ids = set(
    str(x)
    for x in st.session_state.skipped_ids
)

# ============================================================
# FIND UNLABELED MESSAGES
# ============================================================

unlabeled_df = df[
    ~df["message_id"].astype(str).isin(labeled_ids)
    &
    ~df["message_id"].astype(str).isin(skipped_ids)
].copy()

# ============================================================
# NO MESSAGES LEFT
# ============================================================

if unlabeled_df.empty:

    if skipped_ids:

        st.success(
            "🎉 All currently available messages have been reviewed."
        )

        st.info(
            f"{len(skipped_ids)} message(s) were skipped."
        )

        if st.button(
            "🔁 Review Skipped Messages Again",
            use_container_width=True
        ):

            st.session_state.skipped_ids.clear()

            st.rerun()

        st.stop()

    else:

        st.success(
            "🎉 All messages in the dataset have been manually labeled!"
        )

        st.balloons()

        st.stop()

# ============================================================
# CURRENT MESSAGE
# ============================================================

current_message = unlabeled_df.iloc[0]

current_message_id = str(
    current_message["message_id"]
)

message_text = str(
    current_message["text"]
)

original_label = get_original_label(
    current_message
)

# ============================================================
# MESSAGE NUMBER
# ============================================================

st.subheader(
    f"📩 Message #{current_message_id}"
)

st.markdown(
    '<div class="info-box">'
    'Read the SMS carefully and select the correct label.'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# DISPLAY MESSAGE
# ============================================================

# ============================================================
# DISPLAY MESSAGE WITH COPY BUTTON
# ============================================================

st.subheader("📩 SMS Message")

st.code(
    message_text,
    language=None
)

# ============================================================
# ORIGINAL LABEL
# ============================================================

if original_label:

    st.caption(
        f"Original dataset label: `{original_label}`"
    )

# ============================================================
# BUTTONS
# ============================================================

st.subheader("🏷️ Select Label")

col1, col2, col3, col4 = st.columns(4)

# ============================================================
# SCAM
# ============================================================

with col1:

    if st.button(
        "🚨 SCAM",
        key=f"scam_{current_message_id}",
        use_container_width=True,
        type="primary"
    ):

        save_label(
            message_id=current_message_id,
            text=message_text,
            original_label=original_label,
            manual_label="SCAM"
        )

        st.session_state.skipped_ids.discard(
            current_message_id
        )

        st.session_state.current_message_id = None

        st.rerun()

# ============================================================
# GENUINE
# ============================================================

with col2:

    if st.button(
        "✅ GENUINE",
        key=f"genuine_{current_message_id}",
        use_container_width=True
    ):

        save_label(
            message_id=current_message_id,
            text=message_text,
            original_label=original_label,
            manual_label="GENUINE"
        )

        st.session_state.skipped_ids.discard(
            current_message_id
        )

        st.session_state.current_message_id = None

        st.rerun()

# ============================================================
# NON-SCAM SPAM
# ============================================================

with col3:

    if st.button(
        "📢 NON-SCAM SPAM",
        key=f"spam_{current_message_id}",
        use_container_width=True
    ):

        save_label(
            message_id=current_message_id,
            text=message_text,
            original_label=original_label,
            manual_label="NON-SCAM SPAM"
        )

        st.session_state.skipped_ids.discard(
            current_message_id
        )

        st.session_state.current_message_id = None

        st.rerun()

# ============================================================
# SKIP
# ============================================================

with col4:

    if st.button(
        "🔄 SKIP",
        key=f"skip_{current_message_id}",
        use_container_width=True
    ):

        st.session_state.skipped_ids.add(
            current_message_id
        )

        st.session_state.current_message_id = None

        st.rerun()

# ============================================================
# QUEUE INFORMATION
# ============================================================

st.divider()

remaining_count = len(unlabeled_df)

st.write(
    f"📌 **Messages remaining in current queue:** "
    f"{remaining_count}"
)

if skipped_ids:

    st.write(
        f"🔄 **Skipped messages:** "
        f"{len(skipped_ids)}"
    )

# ============================================================
# TARGET STATUS
# ============================================================

st.divider()

st.subheader("🎯 Target Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    if counts["SCAM"] >= TARGET_PER_CLASS:
        st.success(
            f"🚨 SCAM: {counts['SCAM']}/{TARGET_PER_CLASS}"
        )
    else:
        st.info(
            f"🚨 SCAM: {counts['SCAM']}/{TARGET_PER_CLASS}"
        )

with status_col2:

    if counts["GENUINE"] >= TARGET_PER_CLASS:
        st.success(
            f"✅ GENUINE: {counts['GENUINE']}/{TARGET_PER_CLASS}"
        )
    else:
        st.info(
            f"✅ GENUINE: {counts['GENUINE']}/{TARGET_PER_CLASS}"
        )

with status_col3:

    if counts["NON-SCAM SPAM"] >= TARGET_PER_CLASS:
        st.success(
            f"📢 NON-SCAM SPAM: "
            f"{counts['NON-SCAM SPAM']}/{TARGET_PER_CLASS}"
        )
    else:
        st.info(
            f"📢 NON-SCAM SPAM: "
            f"{counts['NON-SCAM SPAM']}/{TARGET_PER_CLASS}"
        )

# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

with st.expander("📂 Dataset Information"):

    st.write(
        f"**Input dataset:** `{INPUT_FILE}`"
    )

    st.write(
        f"**Output file:** `{OUTPUT_FILE}`"
    )

    st.write(
        f"**Total dataset messages:** {len(df)}"
    )

    st.write(
        f"**Already manually labeled:** {total_labeled}"
    )

    st.write(
        f"**Currently skipped:** {len(skipped_ids)}"
    )

# ============================================================
# DOWNLOAD CSV
# ============================================================

if OUTPUT_FILE.exists():

    st.divider()

    try:

        with open(
            OUTPUT_FILE,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download Manual Labels CSV",
                data=file,
                file_name="manual_labels.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:

        st.warning(
            f"Could not prepare download: {e}"
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SMS Scam Detection Using NLP and Machine Learning"
)