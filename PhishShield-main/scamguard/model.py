from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_vectorizer
from .preprocessing import clean_text


def build_model_pipeline() -> Pipeline:
    vectorizer = build_vectorizer()
    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=None,
    )

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )


def prepare_texts(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]

