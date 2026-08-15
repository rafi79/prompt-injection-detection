import sys
import time
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sanitizer import sanitize
from src.keyword_engine import analyze

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def keyword_only_predict(text, use_sanitization: bool, threshold: int = 1):
    t0 = time.perf_counter()
    clean = sanitize(text) if use_sanitization else re.sub(r"\s+", " ", text).strip()
    kw = analyze(clean)
    label = 1 if kw.total_hits >= threshold else 0
    return label, (time.perf_counter() - t0) * 1000.0


def eval_config(name, texts, labels, predict_fn):
    preds, lats = [], []
    for t in texts:
        p, lat = predict_fn(t)
        preds.append(p)
        lats.append(lat)
    preds = np.array(preds)
    labels = np.array(labels)
    fp = int(((preds == 1) & (labels == 0)).sum())
    neg_total = int((labels == 0).sum())
    fpr = fp / neg_total if neg_total else 0.0
    return {
        "configuration": name,
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "latency_ms_mean": float(np.mean(lats)),
        "fpr": fpr,
    }


def main():
    test = pd.read_csv(DATA / "test.csv")
    texts = test.text.tolist()
    labels = test.label.tolist()

    with open(RESULTS / "risk_scorer.pkl", "rb") as f:
        scorer = pickle.load(f)

    from src.risk_scorer import extract_features

    def full_pipeline_predict(t):
        t0 = time.perf_counter()
        clean = sanitize(t)
        kw = analyze(clean)
        feats = extract_features(clean, kw, source_trust=0.0)
        prob = float(scorer.predict_proba(feats.reshape(1, -1))[0])
        label = 1 if prob >= scorer.flag_threshold else 0
        return label, (time.perf_counter() - t0) * 1000.0

    rows = [
        eval_config("Keyword analysis only, no sanitization",
                    texts, labels, lambda t: keyword_only_predict(t, use_sanitization=False)),
        eval_config("+ Sanitization (Stage 1 + Stage 2)",
                    texts, labels, lambda t: keyword_only_predict(t, use_sanitization=True)),
        eval_config("+ Contextual risk scoring (full pipeline)",
                    texts, labels, full_pipeline_predict),
    ]

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "ablation_results.csv", index=False)


if __name__ == "__main__":
    main()
