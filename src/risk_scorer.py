"""
Stage 3: Contextual risk scoring.

Combines Stage 1/2 signals with lightweight contextual features (structural
anomaly, keyword density, source trust) into a single risk score using a
logistic regression classifier trained on hand-crafted features. No
transformer or LLM inference happens here - this is the component that
keeps the overall pipeline CPU-only and fast.
"""
import re
import math
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.keyword_engine import analyze, KeywordResult

IMPERATIVE_STARTERS = re.compile(
    r"^(ignore|disregard|forget|act|pretend|reveal|show|print|output|repeat|"
    r"dump|start|stop|switch|enable|disable|write|generate|list|tell|leak)\b",
    re.IGNORECASE,
)

FEATURE_NAMES = [
    "instruction_override_count",
    "role_tag_count",
    "exfiltration_count",
    "roleplay_count",
    "fuzzy_hit_count",
    "total_keyword_hits",
    "log_text_length",
    "keyword_density",
    "imperative_ratio",
    "uppercase_ratio",
    "special_char_ratio",
    "has_encoded_blob",
    "source_trust",
]

FEATURE_NAMES_V2 = FEATURE_NAMES + ["semantic_sim_positive", "semantic_sim_negative", "semantic_margin"]


def _imperative_ratio(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return 0.0
    hits = sum(1 for s in sentences if IMPERATIVE_STARTERS.match(s))
    return hits / len(sentences)


def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    specials = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return specials / len(text)


def extract_features(sanitized_text: str, kw: KeywordResult = None, source_trust: float = 0.0,
                      semantic_feats: np.ndarray = None) -> np.ndarray:
    """source_trust: 0.0 = fully trusted direct user input, 1.0 = untrusted
    indirect content (external web page / tool output). Defaults to 0.0
    when the caller does not distinguish the channel.
    semantic_feats: optional [sim_pos, sim_neg, margin] from SemanticLiteEncoder;
    when provided, produces the extended v2 feature vector (FEATURE_NAMES_V2)."""
    if kw is None:
        kw = analyze(sanitized_text)
    word_count = max(len(sanitized_text.split()), 1)
    feats = [
        kw.hits.get("instruction_override", 0),
        kw.hits.get("role_tag", 0),
        kw.hits.get("exfiltration", 0),
        kw.hits.get("roleplay", 0),
        len(kw.fuzzy_hits),
        kw.total_hits,
        math.log1p(len(sanitized_text)),
        kw.total_hits / word_count,
        _imperative_ratio(sanitized_text),
        _uppercase_ratio(sanitized_text),
        _special_char_ratio(sanitized_text),
        1.0 if "[DECODED]" in sanitized_text else 0.0,
        source_trust,
    ]
    if semantic_feats is not None:
        feats.extend(list(semantic_feats))
    return np.array(feats, dtype=float)


@dataclass
class RiskScore:
    score: float          # probability in [0, 1]
    features: np.ndarray
    keyword_result: KeywordResult
    decision: str          # "allow" | "flag" | "block"


class RiskScorer:
    """Thin wrapper around a logistic regression classifier trained on the
    hand-crafted feature vector defined above."""

    def __init__(self, flag_threshold: float = 0.4, block_threshold: float = 0.7):
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.scaler = StandardScaler()
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RiskScorer must be fit() before scoring.")
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)[:, 1]

    def score_one(self, sanitized_text: str, source_trust: float = 0.0) -> RiskScore:
        kw = analyze(sanitized_text)
        feats = extract_features(sanitized_text, kw, source_trust)
        prob = float(self.predict_proba(feats.reshape(1, -1))[0])
        if prob >= self.block_threshold:
            decision = "block"
        elif prob >= self.flag_threshold:
            decision = "flag"
        else:
            decision = "allow"
        return RiskScore(score=prob, features=feats, keyword_result=kw, decision=decision)
