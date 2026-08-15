import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sanitizer import sanitize
from src.keyword_engine import analyze
from src.risk_scorer import RiskScorer, extract_features
from src.baselines import TfidfSvmBaseline

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "results"
MODELS.mkdir(exist_ok=True)


def build_feature_matrix(texts):
    feats = []
    for t in texts:
        clean = sanitize(t)
        kw = analyze(clean)
        feats.append(extract_features(clean, kw, source_trust=0.0))
    return np.vstack(feats)


def main():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")

    print(f"Training on {len(train)} examples ({train.label.sum()} pos / {len(train)-train.label.sum()} neg)")

    # --- proposed system: contextual risk scorer ---
    X_train = build_feature_matrix(train.text.tolist())
    y_train = train.label.values
    scorer = RiskScorer()
    scorer.fit(X_train, y_train)

    # pick flag_threshold on validation set to maximize F1 (simple sweep)
    X_val = build_feature_matrix(val.text.tolist())
    y_val = val.label.values
    probs_val = scorer.predict_proba(X_val)

    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (probs_val >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    scorer.flag_threshold = float(best_t)
    print(f"Selected flag_threshold={best_t:.2f} (val F1={best_f1:.4f})")

    with open(MODELS / "risk_scorer.pkl", "wb") as f:
        pickle.dump(scorer, f)

    # --- baseline: TF-IDF + Linear SVM ---
    tfidf_svm = TfidfSvmBaseline()
    tfidf_svm.fit(train.text.tolist(), train.label.tolist())
    with open(MODELS / "tfidf_svm.pkl", "wb") as f:
        pickle.dump(tfidf_svm, f)

    print("Saved trained models to results/risk_scorer.pkl and results/tfidf_svm.pkl")


if __name__ == "__main__":
    main()
