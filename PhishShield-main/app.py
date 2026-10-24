from __future__ import annotations

import streamlit as st

from scamguard import ScamGuard


st.set_page_config(
    page_title="ScamGuard AI",
    page_icon="🛡️",
    layout="centered",
)


def _label_color(label: str, risk: int) -> tuple[str, str]:
    if label == "Safe" and risk < 35:
        return ("#16a34a", "SAFE")
    if risk < 70:
        return ("#ca8a04", "SUSPICIOUS")
    return ("#dc2626", "HIGH RISK")


@st.cache_resource
def get_guard() -> ScamGuard:
    guard = ScamGuard()
    guard.ensure_ready()
    return guard


st.title("ScamGuard AI – Real-Time Scam Detection")
st.caption("For student cybersecurity awareness. If this looks dangerous, verify via official channels.")

message = st.text_area(
    "Paste a message to analyze",
    height=190,
    placeholder="Enter the message text...",
)

analyze = st.button("Analyze", type="primary", use_container_width=True)

if "stats" not in st.session_state:
    st.session_state["stats"] = {"total": 0, "scams": 0, "counts": {}}

if analyze:
    if not message.strip():
        st.warning("Please paste a message first.")
    else:
        out = get_guard().analyze(message)

        st.session_state["stats"]["total"] += 1
        if out.label != "Safe":
            st.session_state["stats"]["scams"] += 1
        st.session_state["stats"]["counts"][out.label] = (
            st.session_state["stats"]["counts"].get(out.label, 0) + 1
        )

        st.subheader("Result")
        color, risk_band = _label_color(out.label, out.risk_score)
        st.markdown(
            f"""
            <div style="display:flex;gap:10px;align-items:center;">
              <div style="padding:6px 10px;border-radius:999px;background:{color};color:white;font-weight:700;">
                {out.label}
              </div>
              <div style="color:#6b7280;font-weight:600;">{risk_band}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(f"Confidence: **{out.confidence_pct}%**")
        st.write(f"Risk score: **{out.risk_score}/100**")
        st.progress(out.risk_score / 100)

        st.subheader("Explainability")
        if out.suspicious_keywords:
            st.write("Suspicious keywords:", ", ".join(out.suspicious_keywords))
        else:
            st.write("Suspicious keywords: None")

        if out.suspicious_phrases:
            st.write("Scam-style phrases:", ", ".join(out.suspicious_phrases))
        else:
            st.write("Scam-style phrases: None")

        st.write("Reason:", out.reason)

        st.subheader("Safety tips")
        for tip in out.tips:
            st.write(f"- {tip}")

        st.subheader("Mini dashboard")
        stats = st.session_state["stats"]
        most_common = None
        if stats["counts"]:
            most_common = max(stats["counts"].items(), key=lambda kv: kv[1])[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages analyzed", stats["total"])
        c2.metric("Scams detected", stats["scams"])
        c3.metric("Most common type", most_common or "—")