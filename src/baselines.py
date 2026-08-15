"""
Baseline detectors used for comparison against the proposed lightweight
pipeline.

B1: Keyword-only  - Stage 2 of the proposed pipeline used in isolation,
    thresholded on raw pattern-match count. Represents the cheapest,
    least contextual approach.

B2: TF-IDF + Linear SVM - a heavier classical-ML text classifier operating
    on a full bag-of-words TF-IDF representation (thousands of features)
    rather than the 13 hand-crafted features used by the proposed system.
    This stands in for a "heavier learned classifier" baseline.

Note on scope: a fine-tuned transformer baseline (e.g. a PromptGuard-style
DistilBERT/RoBERTa classifier) and an LLM-judge baseline are described in
the paper but are NOT executed in this environment, because it has no
network access to Hugging Face Hub for pretrained weights and no
configured LLM API credentials. See benchmark.py for how to plug them in
if you run this on a machine with that access.
"""
import time
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.sanitizer import sanitize
from src.keyword_engine import analyze


class KeywordOnlyBaseline:
    def __init__(self, hit_threshold: int = 1):
        self.hit_threshold = hit_threshold

    def fit(self, texts, labels):
        # thresholding needs no training; kept for a uniform interface
        return self

    def predict_label(self, raw_text: str) -> int:
        clean = sanitize(raw_text)
        kw = analyze(clean)
        return 1 if kw.total_hits >= self.hit_threshold else 0

    def predict_label_timed(self, raw_text: str):
        t0 = time.perf_counter()
        label = self.predict_label(raw_text)
        return label, (time.perf_counter() - t0) * 1000.0


class TfidfSvmBaseline:
    def __init__(self, max_features: int = 20000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), sublinear_tf=True
        )
        self.clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
        self._fitted = False

    def fit(self, texts, labels):
        clean = [sanitize(t) for t in texts]
        X = self.vectorizer.fit_transform(clean)
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def predict_label(self, raw_text: str) -> int:
        clean = sanitize(raw_text)
        X = self.vectorizer.transform([clean])
        return int(self.clf.predict(X)[0])

    def predict_label_timed(self, raw_text: str):
        t0 = time.perf_counter()
        label = self.predict_label(raw_text)
        return label, (time.perf_counter() - t0) * 1000.0
