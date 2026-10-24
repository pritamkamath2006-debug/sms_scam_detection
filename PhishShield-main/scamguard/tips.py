"""
scamguard/tips.py
=================
Practical, actionable cybersecurity tips tailored to 3-class SMS classification.
"""

from __future__ import annotations

TIPS_BY_CLASS: dict[str, list[str]] = {
    "SCAM": [
        "🚨 NEVER share OTPs, PINs, passwords, or CVV numbers with anyone over SMS or phone.",
        "🛑 Do not click on unknown links or download APK files from SMS alerts.",
        "🏦 Banks never ask you to update KYC or unblock accounts via third-party web links.",
        "📞 If you suspect fraud, verify independently through official customer support numbers (never use numbers in the message).",
        "🛡️ Report the fraud SMS to cybercrime authorities (e.g., dial 1930 or visit cybercrime.gov.in in India).",
    ],
    "NON-SCAM SPAM": [
        "📢 Reply with 'STOP' or check if your telecom provider supports DND (Do Not Disturb) registry.",
        "⚠️ Avoid clicking promotional links from unverified brands or senders.",
        "🚫 Do not call premium-rate numbers (e.g. starting with 090 or special tariff codes) advertised in promotional texts.",
        "🗑️ Delete unwanted marketing messages to keep your inbox clutter-free.",
    ],
    "GENUINE": [
        "✅ This message matches regular personal or transactional communication patterns.",
        "💡 Even with genuine-looking alerts, always verify that the sender ID matches your bank or service provider.",
        "🔒 Never share personal details unless you initiated the conversation through an official application.",
    ],
}


def safety_tips(label: str) -> list[str]:
    """Returns a list of actionable safety tips for the predicted class."""
    return TIPS_BY_CLASS.get(label, TIPS_BY_CLASS["GENUINE"])
