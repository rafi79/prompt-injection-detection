import sys
import time
import json
import pickle
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.sanitizer import sanitize
from src.keyword_engine import analyze
from src.risk_scorer import extract_features
from src.pipeline import InjectionDetectionPipeline
from src.baselines import KeywordOnlyBaseline

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_models():
    with open(RESULTS / "risk_scorer.pkl", "rb") as f:
        scorer = pickle.load(f)
    with open(RESULTS / "tfidf_svm.pkl", "rb") as f:
        tfidf_svm = pickle.load(f)
    return scorer, tfidf_svm


def eval_system(name, predict_fn, texts, labels, n_timing_reps=1):
    """predict_fn(text) -> (label:int, latency_ms:float) for a single text."""
    preds = []
    latencies = []
    tracemalloc.start()
    t_start = time.perf_counter()
    for t in texts:
        label, lat_ms = predict_fn(t)
        preds.append(label)
        latencies.append(lat_ms)
    total_wall_s = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    preds = np.array(preds)
    labels = np.array(labels)
    metrics = {
        "system": name,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "peak_memory_mb": peak / (1024 * 1024),
        "throughput_req_s": len(texts) / total_wall_s,
        "n_test": len(texts),
    }
    return metrics


def main():
    test = pd.read_csv(DATA / "test.csv")
    texts = test.text.tolist()
    labels = test.label.tolist()

    scorer, tfidf_svm = load_models()
    pipeline = InjectionDetectionPipeline(scorer)
    kw_baseline = KeywordOnlyBaseline(hit_threshold=1)

    def predict_proposed(t):
        t0 = time.perf_counter()
        clean = sanitize(t)
        kw = analyze(clean)
        feats = extract_features(clean, kw, source_trust=0.0)
        prob = float(scorer.predict_proba(feats.reshape(1, -1))[0])
        label = 1 if prob >= scorer.flag_threshold else 0
        return label, (time.perf_counter() - t0) * 1000.0

    def predict_keyword_only(t):
        return kw_baseline.predict_label_timed(t)

    def predict_tfidf_svm(t):
        return tfidf_svm.predict_label_timed(t)

    results = []
    results.append(eval_system("Proposed (Sanitize+Keyword+RiskScore)", predict_proposed, texts, labels))
    results.append(eval_system("B1: Keyword-only", predict_keyword_only, texts, labels))
    results.append(eval_system("B2: TF-IDF + Linear SVM", predict_tfidf_svm, texts, labels))

    df = pd.DataFrame(results)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "benchmark_results.csv", index=False)
    with open(RESULTS / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS / 'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
