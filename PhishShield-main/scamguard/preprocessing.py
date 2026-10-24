from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean and normalize text for the classifier.

    Requirements:
    - Convert to lowercase
    - Remove noise and extra spaces
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\u200b", " ")  # zero-width space
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text

