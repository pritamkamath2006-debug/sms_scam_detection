from __future__ import annotations


TIPS_BY_CLASS: dict[str, list[str]] = {
    "Safe": [
        "Still be cautious with unknown senders.",
        "Avoid sharing personal information unnecessarily.",
    ],
    "Phishing": [
        "Do not click links in unexpected messages.",
        "Verify the sender via an official website/app (not the message link).",
        "Never share passwords or banking PINs.",
    ],
    "OTP Scam": [
        "Never share OTP/verification codes with anyone.",
        "If you receive an OTP you didn’t request, secure your account immediately.",
        "Enable strong 2FA and review recent login activity.",
    ],
    "Lottery Scam": [
        "Be skeptical of 'you won' messages you didn’t enter.",
        "Never pay fees to claim a prize.",
        "Verify by contacting the organization via official channels.",
    ],
    "Job Scam": [
        "A real employer won’t ask you to pay to get hired.",
        "Avoid moving to Telegram/WhatsApp for 'interviews' immediately.",
        "Verify the company and recruiter on official sites and LinkedIn.",
    ],
}


def safety_tips(label: str) -> list[str]:
    return TIPS_BY_CLASS.get(label, TIPS_BY_CLASS["Safe"])

