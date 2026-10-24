from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    """
    Builds a TF-IDF Vectorizer with unigrams and bigrams,
    sublinear TF scaling, and lowercase normalization.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
