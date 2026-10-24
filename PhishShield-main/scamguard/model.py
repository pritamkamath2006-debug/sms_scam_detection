"""
scamguard/model.py
==================
Scikit-learn model pipeline builders for 3-class SMS classification.
"""

from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from .features import build_vectorizer
from .preprocessing import clean_text


def build_model_pipeline(model_type: str = "linear_svc") -> Pipeline:
    """
    Returns an end-to-end TF-IDF + Classifier Pipeline.
    Supports:
      - 'linear_svc' (default, best baseline)
      - 'logistic_regression'
      - 'naive_bayes'
      - 'random_forest'
    """
    vectorizer = build_vectorizer()

    if model_type == "linear_svc":
        clf = LinearSVC(class_weight="balanced", random_state=42)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    elif model_type == "naive_bayes":
        clf = MultinomialNB()
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])


def prepare_texts(texts: list[str]) -> list[str]:
    """Applies standardized cleaning across all input texts."""
    return [clean_text(t) for t in texts]
