import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sanitizer import sanitize
from src.keyword_engine import analyze
from src.risk_scorer import RiskScorer, extract_features
from src.semantic_lite import SemanticLiteEncoder

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def build_v1_matrix(texts):
    feats = []
    for t in texts:
        clean = sanitize(t)
        kw = analyze(clean)
        feats.append(extract_features(clean, kw, source_trust=0.0))
    return np.vstack(feats)


def build_v2_matrix(texts, encoder: SemanticLiteEncoder):
    clean_texts = [sanitize(t) for t in texts]
    sem_feats = encoder.similarity_features(clean_texts)
    feats = []
    for clean, sem in zip(clean_texts, sem_feats):
        kw = analyze(clean)
        feats.append(extract_features(clean, kw, source_trust=0.0, semantic_feats=sem))
    return np.vstack(feats)


def tune_threshold(scorer, X_val, y_val):
    probs = scorer.predict_proba(X_val)
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    scorer.flag_threshold = float(best_t)
    return best_t, best_f1


def eval_scorer(name, scorer, X_test, y_test, texts_for_latency, predict_fn_timed):
    probs = scorer.predict_proba(X_test)
    preds = (probs >= scorer.flag_threshold).astype(int)

    lats = []
    for t in texts_for_latency:
        _, lat = predict_fn_timed(t)
        lats.append(lat)

    return {
        "system": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "latency_ms_mean": float(np.mean(lats)),
    }


def main():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    test = pd.read_csv(DATA / "test.csv")

    y_train, y_val, y_test = train.label.values, val.label.values, test.label.values

    # ---- v1: lexical-only (existing pipeline) ----
    X_train_v1 = build_v1_matrix(train.text.tolist())
    X_val_v1 = build_v1_matrix(val.text.tolist())
    X_test_v1 = build_v1_matrix(test.text.tolist())

    scorer_v1 = RiskScorer()
    scorer_v1.fit(X_train_v1, y_train)
    tune_threshold(scorer_v1, X_val_v1, y_val)

    def predict_v1_timed(t):
        t0 = time.perf_counter()
        clean = sanitize(t)
        kw = analyze(clean)
        feats = extract_features(clean, kw, source_trust=0.0)
        prob = float(scorer_v1.predict_proba(feats.reshape(1, -1))[0])
        label = 1 if prob >= scorer_v1.flag_threshold else 0
        return label, (time.perf_counter() - t0) * 1000.0

    # ---- v2: lexical + semantic-lite ----
    encoder = SemanticLiteEncoder(n_components=100)
    encoder.fit(train.text.tolist(), y_train)

    X_train_v2 = build_v2_matrix(train.text.tolist(), encoder)
    X_val_v2 = build_v2_matrix(val.text.tolist(), encoder)
    X_test_v2 = build_v2_matrix(test.text.tolist(), encoder)

    scorer_v2 = RiskScorer()
    scorer_v2.fit(X_train_v2, y_train)
    tune_threshold(scorer_v2, X_val_v2, y_val)

    def predict_v2_timed(t):
        t0 = time.perf_counter()
        clean = sanitize(t)
        kw = analyze(clean)
        sem = encoder.similarity_features_one(clean)
        feats = extract_features(clean, kw, source_trust=0.0, semantic_feats=sem)
        prob = float(scorer_v2.predict_proba(feats.reshape(1, -1))[0])
        label = 1 if prob >= scorer_v2.flag_threshold else 0
        return label, (time.perf_counter() - t0) * 1000.0

    results = [
        eval_scorer("v1: Lexical-only (Sanitize+Keyword+RiskScore)", scorer_v1, X_test_v1, y_test,
                    test.text.tolist(), predict_v1_timed),
        eval_scorer("v2: Lexical + Semantic-lite (TF-IDF+SVD)", scorer_v2, X_test_v2, y_test,
                    test.text.tolist(), predict_v2_timed),
    ]

    df = pd.DataFrame(results)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "semantic_lite_comparison.csv", index=False)

    with open(RESULTS / "risk_scorer_v2.pkl", "wb") as f:
        pickle.dump({"scorer": scorer_v2, "encoder": encoder}, f)

    print(f"\nSaved comparison to {RESULTS / 'semantic_lite_comparison.csv'}")
    print(f"Saved v2 model to {RESULTS / 'risk_scorer_v2.pkl'}")


if __name__ == "__main__":
    main()
