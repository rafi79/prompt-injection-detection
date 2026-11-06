"""Metrics and statistical tests shared by all experiment scripts."""
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    m = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "n": int(len(y)), "n_pos": int(y.sum()),
    }
    if score is not None and len(set(y.tolist())) == 2:
        try:
            m["roc_auc"] = roc_auc_score(y, score)
        except ValueError:
            m["roc_auc"] = float("nan")
    return m


def t_ci95(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return 0.0
    return float(stats.sem(v) * stats.t.ppf(0.975, len(v) - 1))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(y, pred_a, pred_b):
    """Exact (binomial) McNemar test on per-example correctness.
    Returns (n_a_right_b_wrong, n_a_wrong_b_right, p_value)."""
    y = np.asarray(y)
    ca, cb = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))
    return b, c, p


def paired_bootstrap_f1(y, pred_a, pred_b, n_boot=2000, seed=0):
    """Bootstrap distribution of F1(a) - F1(b) on the same resampled test
    items. Returns (observed diff, 95% CI low, high, one-sided p(diff<=0))."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    obs = f1_score(y, a, zero_division=0) - f1_score(y, b, zero_division=0)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = f1_score(y[idx], a[idx], zero_division=0) - f1_score(y[idx], b[idx], zero_division=0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), float(lo), float(hi), float((diffs <= 0).mean())


def paired_t(values_a, values_b):
    """Paired t-test across seeds (e.g. per-seed F1 of two systems)."""
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    if len(a) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_rel(a, b)
    return float(np.mean(a - b)), float(p)
