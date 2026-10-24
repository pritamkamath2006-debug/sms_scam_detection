"""
manual_labeling.py
==================
Interactive Streamlit application for human-in-the-loop 3-class SMS labeling.
Labels messages into:
  - SCAM
  - GENUINE
  - NON-SCAM SPAM
Supports SKIP and prevents duplicate reviews by matching text content.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMS Manual Labeling Studio",
    page_icon="🏷️",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

INPUT_FILE = DATASET_DIR / "scam_training_pool.csv"
OUTPUT_FILE = DATASET_DIR / "scam_pool_labels.csv"
REVIEWED_FILE = DATASET_DIR / "reviewed_dataset.csv"

LABELS = ["SCAM", "GENUINE", "NON-SCAM SPAM"]

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "skipped_texts" not in st.session_state:
    st.session_state.skipped_texts = set()

# ============================================================
# LOAD CANDIDATES & LABELS
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        st.error(f"Candidate file not found: {INPUT_FILE}")
        return pd.DataFrame(), pd.DataFrame()

    pool_df = pd.read_csv(INPUT_FILE)
    pool_df["text"] = pool_df["text"].fillna("").astype(str).str.strip()
    pool_df = pool_df[pool_df["text"] != ""].copy()

    # Load existing labels
    if OUTPUT_FILE.exists():
        labels_df = pd.read_csv(OUTPUT_FILE)
    else:
        labels_df = pd.DataFrame(columns=["message_id", "text", "original_label", "manual_label"])

    reviewed_texts = set()
    if REVIEWED_FILE.exists():
        try:
            r_df = pd.read_csv(REVIEWED_FILE)
            for t in r_df["text"].dropna():
                reviewed_texts.add(t.strip().lower())
        except Exception:
            pass

    for t in labels_df["text"].dropna():
        reviewed_texts.add(t.strip().lower())

    return pool_df, labels_df, reviewed_texts


def save_label(message_id, text, orig_label, chosen_label):
    if OUTPUT_FILE.exists():
        labels_df = pd.read_csv(OUTPUT_FILE)
    else:
        labels_df = pd.DataFrame(columns=["message_id", "text", "original_label", "manual_label"])

    # Remove prior duplicate of this exact text
    labels_df = labels_df[labels_df["text"].str.strip().str.lower() != text.strip().lower()].copy()

    new_row = pd.DataFrame([{
        "message_id": str(message_id),
        "text": text,
        "original_label": orig_label,
        "manual_label": chosen_label
    }])

    updated = pd.concat([labels_df, new_row], ignore_index=True)
    updated.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")


# ============================================================
# MAIN UI
# ============================================================

st.title("🏷️ SMS 3-Class Manual Labeling Studio")
st.caption("Review candidate messages into: SCAM, GENUINE, or NON-SCAM SPAM")

pool_df, labels_df, reviewed_texts = load_data()

if pool_df.empty:
    st.warning("No candidate messages found.")
    st.stop()

# Filter candidates not yet reviewed and not skipped in this session
unlabeled_df = pool_df[
    (~pool_df["text"].str.strip().str.lower().isin(reviewed_texts)) &
    (~pool_df["text"].isin(st.session_state.skipped_texts))
].copy().reset_index(drop=True)

# Metrics bar
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Candidates", len(pool_df))
m2.metric("Already Reviewed", len(reviewed_texts))
m3.metric("Remaining", len(unlabeled_df))
m4.metric("Session Skipped", len(st.session_state.skipped_texts))

st.divider()

if unlabeled_df.empty:
    st.success("🎉 All available candidate messages in this queue have been reviewed or skipped!")
    if st.button("Reset Skipped Messages"):
        st.session_state.skipped_texts.clear()
        st.rerun()
else:
    current_row = unlabeled_df.iloc[0]
    msg_id = current_row.get("message_id", "")
    msg_text = current_row.get("text", "")
    orig_label = current_row.get("label", "unknown")

    st.subheader(f"Message ID: {msg_id} (Original: `{orig_label}`)")

    st.info(f"**SMS Text:**\n\n> {msg_text}")

    st.write("### Assign Category:")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("🚨 SCAM", type="primary", use_container_width=True):
            save_label(msg_id, msg_text, orig_label, "SCAM")
            st.rerun()

    with c2:
        if st.button("✅ GENUINE", use_container_width=True):
            save_label(msg_id, msg_text, orig_label, "GENUINE")
            st.rerun()

    with c3:
        if st.button("📢 NON-SCAM SPAM", use_container_width=True):
            save_label(msg_id, msg_text, orig_label, "NON-SCAM SPAM")
            st.rerun()

    with c4:
        if st.button("🔄 SKIP (Uncertain)", use_container_width=True):
            st.session_state.skipped_texts.add(msg_text)
            st.rerun()

st.divider()
with st.expander("📊 View Current Label Distribution in scam_pool_labels.csv"):
    if OUTPUT_FILE.exists():
        curr_df = pd.read_csv(OUTPUT_FILE)
        st.write(curr_df["manual_label"].value_counts())
        st.dataframe(curr_df.tail(15))
    else:
        st.write("No labels saved yet.")
