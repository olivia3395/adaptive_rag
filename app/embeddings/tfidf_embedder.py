from __future__ import annotations

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TfidfEmbedder:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.is_fit = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        mat = self.vectorizer.fit_transform(texts)
        self.is_fit = True
        return mat

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Embedder must be fit before transform.")
        return self.vectorizer.transform(texts)
