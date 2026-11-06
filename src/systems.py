"""
Every detector evaluated in the paper behind ONE interface, so that every
experiment (multi-seed, leave-one-source-out, obfuscation, paraphrase,
latency) trains, thresholds and scores them identically.

    det.fit(train_texts, train_labels, val_texts, val_labels)
    det.scores(texts)  -> np.ndarray of risk scores (vectorised where possible)
    det.score_one(text) -> float    (single-request path, used for latency)
    det.predict(texts) -> 0/1 labels using det.threshold

Thresholds: every learned system picks its decision threshold on the
validation split by maximising F1 over the same grid (0.02..0.98). B1 has no
training and flags any text with >= 1 keyword/pattern hit.

The PromptGuard baseline (B3) lives in src/promptguard.py (needs torch).
"""
import time
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.sanitizer import sanitize
from src.keyword_engine import analyze
from src.risk_scorer import extract_features, FEATURE_NAMES

# The learned risk scorer uses 11 of the 13 hand-crafted features.
# Excluded: "has_encoded_blob" (non-zero for only a handful of training rows,
# so its standardised value explodes and its weight is unidentifiable - it
# made ANY decoded payload, even benign, look malicious) and "source_trust"
# (constant 0.0 for every direct-input example). Decoded payloads are still
# analysed by the keyword and semantic stages.
EXCLUDED_FEATURES = ("has_encoded_blob", "source_trust")
LEX_KEEP = [i for i, f in enumerate(FEATURE_NAMES) if f not in EXCLUDED_FEATURES]
from src.semantic_lite import SemanticLiteEncoder, crossfit_features

THRESHOLD_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def tune_threshold(scores, labels, grid=THRESHOLD_GRID):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


class Detector:
    name = "base"
    threshold = 0.5

    def fit(self, tr_x, tr_y, va_x, va_y):
        raise NotImplementedError

    def scores(self, texts):
        return np.array([self.score_one(t) for t in texts])

    def score_one(self, text):
        raise NotImplementedError

    def predict(self, texts):
        return (self.scores(texts) >= self.threshold).astype(int)

    def _tune(self, va_x, va_y):
        self.threshold, self.val_f1 = tune_threshold(self.scores(va_x), np.asarray(va_y))
        return self


# ------------------------------------------------------------------ B1 -----
class KeywordOnly(Detector):
    """B1: flag if the keyword/pattern engine fires at least once."""

    def __init__(self, sanitizer_mode="full", hit_threshold=1):
        self.name = "B1"
        self.sanitizer_mode = sanitizer_mode
        self.threshold = hit_threshold

    def fit(self, *a):
        return self

    def score_one(self, text):
        return float(analyze(sanitize(text, self.sanitizer_mode)).total_hits)


# ------------------------------------------------------------- v1 / v2 -----
class RiskScorerDetector(Detector):
    """v1 (lexical, 13 features) and v2/v2-cf/v3 (+ semantic-lite features).

    semantic=None                        -> v1
    semantic=SemanticLiteEncoder, crossfit=False -> original v2 (in-sample)
    semantic=SemanticLiteEncoder, crossfit=True  -> cross-fitted variant
    """

    def __init__(self, name, semantic: SemanticLiteEncoder = None, crossfit=False,
                 sanitizer_mode="full", seed=42):
        self.name = name
        self.semantic = semantic
        self.crossfit = crossfit
        self.sanitizer_mode = sanitizer_mode
        self.seed = seed

    def _lex(self, clean):
        return extract_features(clean, analyze(clean), source_trust=0.0)[LEX_KEEP]

    def _matrix(self, texts, sem=None):
        clean = [sanitize(t, self.sanitizer_mode) for t in texts]
        lex = np.vstack([self._lex(c) for c in clean])
        if self.semantic is None:
            return lex, clean
        if sem is None:
            sem = self.semantic.similarity_features(clean)
        return np.hstack([lex, sem]), clean

    def fit(self, tr_x, tr_y, va_x, va_y):
        tr_y = np.asarray(tr_y)
        sem_tr = None
        if self.semantic is not None:
            clean_tr = [sanitize(t, self.sanitizer_mode) for t in tr_x]
            if self.crossfit:
                sem_tr = crossfit_features(self.semantic, clean_tr, tr_y, seed=self.seed)
            self.semantic.fit(clean_tr, tr_y)
            if not self.crossfit:
                sem_tr = self.semantic.similarity_features(clean_tr)
        X, _ = self._matrix(tr_x, sem_tr)
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            self.scaler.transform(X), tr_y)
        return self._tune(va_x, va_y)

    def scores(self, texts):
        X, _ = self._matrix(list(texts))
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def score_one(self, text):
        clean = sanitize(text, self.sanitizer_mode)
        f = self._lex(clean)
        if self.semantic is not None:
            f = np.concatenate([f, self.semantic.similarity_features_one(clean)])
        return float(self.model.predict_proba(self.scaler.transform(f.reshape(1, -1)))[0, 1])

    def score_one_staged(self, text):
        """Per-stage wall-clock breakdown (ms) for one request."""
        t0 = time.perf_counter()
        clean = sanitize(text, self.sanitizer_mode)
        t1 = time.perf_counter()
        kw = analyze(clean)
        t2 = time.perf_counter()
        f = extract_features(clean, kw, source_trust=0.0)[LEX_KEEP]
        t3 = time.perf_counter()
        if self.semantic is not None:
            f = np.concatenate([f, self.semantic.similarity_features_one(clean)])
        t4 = time.perf_counter()
        p = float(self.model.predict_proba(self.scaler.transform(f.reshape(1, -1)))[0, 1])
        t5 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1000.0
        return p, {"sanitize": ms(t0, t1), "keyword": ms(t1, t2), "lexical_features": ms(t2, t3),
                   "semantic": ms(t3, t4), "classifier": ms(t4, t5)}


# ------------------------------------------------------------------ B2 -----
class TfidfSvm(Detector):
    """B2: word 1-2gram TF-IDF (20k) + calibrated linear SVM."""

    def __init__(self, sanitizer_mode="full", max_features=20000):
        self.name = "B2"
        self.sanitizer_mode = sanitizer_mode
        self.max_features = max_features

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.vec = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), sublinear_tf=True)
        X = self.vec.fit_transform([sanitize(t, self.sanitizer_mode) for t in tr_x])
        self.clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3).fit(X, np.asarray(tr_y))
        return self._tune(va_x, va_y)

    def scores(self, texts):
        X = self.vec.transform([sanitize(t, self.sanitizer_mode) for t in texts])
        return self.clf.predict_proba(X)[:, 1]

    def score_one(self, text):
        X = self.vec.transform([sanitize(text, self.sanitizer_mode)])
        return float(self.clf.predict_proba(X)[0, 1])


# ---------------------------------------------------- shortcut diagnostic --
class LengthOnly(Detector):
    """Diagnostic: logistic regression on log(character length) only. If this
    scores well, the dataset contains a length shortcut."""

    def __init__(self):
        self.name = "LenOnly"

    @staticmethod
    def _f(texts):
        return np.array([[math.log1p(len(str(t)))] for t in texts])

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.m = LogisticRegression(class_weight="balanced").fit(self._f(tr_x), np.asarray(tr_y))
        return self._tune(va_x, va_y)

    def scores(self, texts):
        return self.m.predict_proba(self._f(texts))[:, 1]

    def score_one(self, text):
        return float(self.scores([text])[0])


# ----------------------------------------------- windowed max-pooling -----
import re as _re


def text_windows(text, window_chars=400):
    """Overlapping windows of consecutive sentences (~window_chars each,
    stride of about half a window)."""
    sents = [s for s in _re.split(r"(?<=[.!?\n])\s+", str(text)) if s.strip()]
    if len(text) <= window_chars or len(sents) <= 1:
        return []
    wins, i = [], 0
    while i < len(sents):
        cur, j = "", i
        while j < len(sents) and (len(cur) < window_chars or cur == ""):
            cur = (cur + " " + sents[j]).strip()
            j += 1
        wins.append(cur)
        if j >= len(sents):
            break
        step = max(1, (j - i) // 2)
        i += step
    return wins


class WindowedDetector(Detector):
    """Mitigation for dilution / benign-padding attacks: score the full text
    AND every sentence window; risk = max. Wraps an already-trained detector;
    `fit` only re-tunes the threshold on validation."""

    def __init__(self, base: Detector, window_chars=400):
        self.base = base
        self.window_chars = window_chars
        self.name = f"{base.name}+win"

    def fit(self, tr_x, tr_y, va_x, va_y):
        return self._tune(va_x, va_y)

    def scores(self, texts):
        texts = list(texts)
        allw, owner = [], []
        for i, t in enumerate(texts):
            for w in [t] + text_windows(t, self.window_chars):
                allw.append(w)
                owner.append(i)
        s = self.base.scores(allw)
        out = np.full(len(texts), -np.inf)
        np.maximum.at(out, np.array(owner), s)
        return out

    def score_one(self, text):
        return float(self.scores([text])[0])


# ------------------------------------------------------------- factory -----
CPU_SYSTEMS = ["B1", "v1", "v2", "v2cf", "v3", "B2"]
DIAGNOSTIC_SYSTEMS = ["LenOnly"]

DESCRIPTIONS = {
    "B1": "Keyword-only",
    "v1": "Lexical risk scorer (11 learned lexical/structural features)",
    "v2": "v1 + semantic-lite (original: word LSA, centroids, in-sample)",
    "v2cf": "v2 with cross-fitted (out-of-fold) semantic features",
    "v3": "v1 + semantic-lite (word+char LSA, 8 prototypes/class, kNN-10, cross-fitted)",
    "B2": "TF-IDF + calibrated linear SVM",
    "B3": "PromptGuard-86M fine-tuned (all parameters)",
    "B3raw": "PromptGuard-86M out-of-the-box",
    "LenOnly": "Length-only logistic regression (shortcut diagnostic)",
}


def make_system(name, seed=42, sanitizer_mode="full"):
    if name == "B1":
        return KeywordOnly(sanitizer_mode)
    if name == "v1":
        return RiskScorerDetector("v1", None, sanitizer_mode=sanitizer_mode, seed=seed)
    if name == "v2":
        return RiskScorerDetector("v2", SemanticLiteEncoder(random_state=seed), crossfit=False,
                                  sanitizer_mode=sanitizer_mode, seed=seed)
    if name == "v2cf":
        return RiskScorerDetector("v2cf", SemanticLiteEncoder(random_state=seed), crossfit=True,
                                  sanitizer_mode=sanitizer_mode, seed=seed)
    if name == "v3":
        enc = SemanticLiteEncoder(analyzer="both", n_prototypes=8, knn=10, random_state=seed)
        return RiskScorerDetector("v3", enc, crossfit=True, sanitizer_mode=sanitizer_mode, seed=seed)
    if name.startswith("sem:"):
        # ablation hook, e.g. "sem:char:8:0:cf"  -> analyzer, prototypes, knn, crossfit
        _, analyzer, k, knn, cf = name.split(":")
        enc = SemanticLiteEncoder(analyzer=analyzer, n_prototypes=int(k), knn=int(knn), random_state=seed)
        return RiskScorerDetector(name, enc, crossfit=(cf == "cf"), sanitizer_mode=sanitizer_mode, seed=seed)
    if name == "B2":
        return TfidfSvm(sanitizer_mode)
    if name == "LenOnly":
        return LengthOnly()
    if name in ("B3", "B3raw"):
        from src.promptguard import PromptGuardDetector
        return PromptGuardDetector(finetune=(name == "B3"), seed=seed)
    raise ValueError(name)
