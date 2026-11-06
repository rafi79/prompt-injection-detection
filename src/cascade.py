"""
Latency-aware two-stage cascade.

Stage 1 is a cheap CPU detector (B2 or v3). It decides on its own when its
score is confidently low (< lo) or confidently high (>= hi). Only inputs in
the uncertainty band [lo, hi) are escalated to stage 2 (PromptGuard-86M,
fine-tuned), whose own validation threshold then decides.

The band is chosen on the VALIDATION split: among all (lo, hi) whose
escalation rate on validation is <= max_escalation, pick the one with the
highest validation F1.

Optionally the text passed to stage 2 is the sanitized text (decoded
payloads appended), so the transformer also sees what an encoding hid.
"""
import numpy as np
from sklearn.metrics import f1_score

from src.systems import Detector
from src.sanitizer import sanitize

BAND_GRID = np.round(np.arange(0.0, 1.0001, 0.02), 2)


def choose_band(s1, s2_pred, y, max_escalation):
    """s1: stage-1 scores, s2_pred: stage-2 0/1 decisions (same items),
    y: labels. Returns (lo, hi, f1, escalation_rate)."""
    s1, s2_pred, y = np.asarray(s1), np.asarray(s2_pred), np.asarray(y)
    best = None
    for i, lo in enumerate(BAND_GRID):
        for hi in BAND_GRID[i:]:
            esc = (s1 >= lo) & (s1 < hi)
            rate = esc.mean()
            if rate > max_escalation + 1e-9:
                continue
            pred = np.where(esc, s2_pred, (s1 >= hi).astype(int))
            f = f1_score(y, pred, zero_division=0)
            key = (f, -rate)
            if best is None or key > best[0]:
                best = (key, float(lo), float(hi), float(f), float(rate))
    return best[1], best[2], best[3], best[4]


def cascade_predict(s1, s2_pred, lo, hi):
    s1 = np.asarray(s1)
    esc = (s1 >= lo) & (s1 < hi)
    return np.where(esc, np.asarray(s2_pred), (s1 >= hi).astype(int)), esc


def _centre(s, t):
    """Monotone map of a score s in [0,1] so that its threshold t maps to 0.5.
    Keeps a continuous risk score (the adaptive attacker sees a real score,
    not a 0/1 decision) while predict() still thresholds at 0.5."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    t = float(np.clip(t, 1e-6, 1 - 1e-6))
    return np.where(s < t, 0.5 * s / t, 0.5 + 0.5 * (s - t) / (1 - t))


class CascadeDetector(Detector):
    """Wraps two TRAINED detectors. fit() only chooses the band on validation."""

    def __init__(self, first, second, max_escalation=0.2, sanitize_second=False):
        self.first, self.second = first, second
        self.max_escalation = max_escalation
        self.sanitize_second = sanitize_second
        tag = "+san" if sanitize_second else ""
        self.name = f"{first.name}>{second.name}{tag}@{int(round(max_escalation * 100))}"
        self.threshold = 0.5
        self.last_escalation_rate = np.nan

    def _second_input(self, texts):
        return [sanitize(t) for t in texts] if self.sanitize_second else list(texts)

    def fit(self, tr_x, tr_y, va_x, va_y):
        va_x = list(va_x)
        s1 = self.first.scores(va_x)
        s2 = self.second.predict(self._second_input(va_x))
        self.lo, self.hi, self.val_f1, self.val_escalation = choose_band(s1, s2, va_y, self.max_escalation)
        return self

    def scores(self, texts):
        """Continuous risk score; >= 0.5 exactly when the cascade flags."""
        texts = list(texts)
        s1 = np.asarray(self.first.scores(texts), dtype=float)
        esc = (s1 >= self.lo) & (s1 < self.hi)
        out = np.where(s1 >= self.hi, _centre(s1, self.hi), _centre(s1, max(self.lo, 1e-6)) * (s1 < self.lo))
        out = np.where(s1 < self.lo, _centre(s1, max(self.lo, 1e-6)), out)
        if esc.any():
            idx = np.where(esc)[0]
            s2 = np.asarray(self.second.scores(self._second_input([texts[i] for i in idx])), dtype=float)
            out[idx] = _centre(s2, self.second.threshold)
        self.last_escalation_rate = float(esc.mean()) if len(texts) else np.nan
        return out

    def score_one(self, text):
        s1 = float(self.first.score_one(text))
        if s1 >= self.hi:
            return float(_centre(s1, self.hi))
        if s1 < self.lo:
            return float(_centre(s1, max(self.lo, 1e-6)))
        t = sanitize(text) if self.sanitize_second else text
        return float(_centre(self.second.score_one(t), self.second.threshold))
