"""
Cross-source generalization test.

The main dataset mixes all three sources into train/val/test, so a model
could in principle overfit to source-specific writing style. This script
retrains with the pr1m8 categorized examples held out ENTIRELY (never seen
in train or val), then evaluates on that held-out set. Because pr1m8 comes
with category/subcategory labels, this simultaneously gives an honest
out-of-distribution generalization number and a per-attack-category recall
breakdown that the original dataset didn't support.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.sanitizer import sanitize
from src.keyword_engine import analyze
from src.risk_scorer import RiskScorer, extract_features
from src.semantic_lite import SemanticLiteEncoder

RAW = ROOT / "data" / "raw"
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SEED = 42


def load_jailbreak_llms():
    jb = pd.read_csv(RAW / "jailbreak_llms_prompts" / "jailbreak_prompts_2023_12_25.csv")
    reg = pd.read_csv(RAW / "jailbreak_llms_prompts" / "regular_prompts_2023_12_25.csv")
    jb_df = pd.DataFrame({"text": jb["prompt"].astype(str), "label": 1})
    reg_df = pd.DataFrame({"text": reg["prompt"].astype(str), "label": 0})
    return jb_df, reg_df


def load_deepset():
    tr = pd.read_parquet(RAW / "deepset_train.parquet")
    te = pd.read_parquet(RAW / "deepset_test.parquet")
    df = pd.concat([tr, te], ignore_index=True)
    return df[df.label == 1][["text", "label"]], df[df.label == 0][["text", "label"]]


def build_features(texts, encoder):
    clean_texts = [sanitize(t) for t in texts]
    sem_feats = encoder.similarity_features(clean_texts)
    feats = []
    for clean, sem in zip(clean_texts, sem_feats):
        kw = analyze(clean)
        feats.append(extract_features(clean, kw, source_trust=0.0, semantic_feats=sem))
    return np.vstack(feats)


def main():
    # --- build train/val WITHOUT pr1m8 at all ---
    jb_pos, jb_neg = load_jailbreak_llms()
    ds_pos, ds_neg = load_deepset()
    positives = pd.concat([jb_pos, ds_pos], ignore_index=True)
    negatives = pd.concat([jb_neg, ds_neg], ignore_index=True)

    for df in (positives, negatives):
        df["text"] = df["text"].astype(str).str.strip()
    positives = positives[positives.text.str.len() > 5].drop_duplicates(subset="text")
    negatives = negatives[negatives.text.str.len() > 5].drop_duplicates(subset="text")

    n_pos = len(positives)
    negatives = negatives.sample(n=min(len(negatives), n_pos * 3), random_state=SEED)

    pos = positives.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    neg = negatives.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    def split(df):
        n = len(df)
        n_train = int(n * 0.80)
        return df.iloc[:n_train], df.iloc[n_train:]

    pos_tr, pos_va = split(pos)
    neg_tr, neg_va = split(neg)
    train = pd.concat([pos_tr, neg_tr]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val = pd.concat([pos_va, neg_va]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"Generalization train (no pr1m8): {len(train)} ({train.label.sum()} pos / {len(train)-train.label.sum()} neg)")
    print(f"Generalization val   (no pr1m8): {len(val)} ({val.label.sum()} pos / {len(val)-val.label.sum()} neg)")

    # --- train v2 pipeline on this reduced (still large) dataset ---
    encoder = SemanticLiteEncoder(n_components=100)
    encoder.fit(train.text.tolist(), train.label.values)

    X_train = build_features(train.text.tolist(), encoder)
    X_val = build_features(val.text.tolist(), encoder)

    scorer = RiskScorer()
    scorer.fit(X_train, train.label.values)

    probs_val = scorer.predict_proba(X_val)
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (probs_val >= t).astype(int)
        f1 = f1_score(val.label.values, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    scorer.flag_threshold = float(best_t)
    print(f"Selected threshold={best_t:.2f} (in-distribution val F1={best_f1:.4f})")

    # --- evaluate on in-distribution val (sanity check) ---
    def predict(texts):
        X = build_features(texts, encoder)
        probs = scorer.predict_proba(X)
        return (probs >= scorer.flag_threshold).astype(int)

    val_preds = predict(val.text.tolist())
    print("\nIn-distribution validation performance (jailbreak_llms + deepset only):")
    print(f"  Acc={accuracy_score(val.label, val_preds):.4f} "
          f"Prec={precision_score(val.label, val_preds, zero_division=0):.4f} "
          f"Rec={recall_score(val.label, val_preds, zero_division=0):.4f} "
          f"F1={f1_score(val.label, val_preds, zero_division=0):.4f}")

    # --- evaluate on fully held-out pr1m8 (out-of-distribution, categorized) ---
    pr1m8 = pd.read_csv(RAW / "pr1m8_prompt_injections.csv")
    pr1m8["text"] = pr1m8["text"].astype(str).str.strip()
    pr1m8 = pr1m8[pr1m8.text.str.len() > 5]
    pr1m8_labels = np.ones(len(pr1m8), dtype=int)  # all pr1m8 rows are attacks

    pr1m8_preds = predict(pr1m8.text.tolist())
    ood_recall = recall_score(pr1m8_labels, pr1m8_preds)
    print(f"\nOut-of-distribution (pr1m8, fully unseen source, n={len(pr1m8)}):")
    print(f"  Recall={ood_recall:.4f}  ({int(pr1m8_preds.sum())}/{len(pr1m8)} detected)")

    # --- per-category breakdown ---
    pr1m8 = pr1m8.reset_index(drop=True)
    pr1m8["predicted"] = pr1m8_preds
    cat_breakdown = pr1m8.groupby("category").agg(
        n=("predicted", "size"),
        detected=("predicted", "sum"),
    )
    cat_breakdown["recall"] = cat_breakdown["detected"] / cat_breakdown["n"]
    cat_breakdown = cat_breakdown.sort_values("recall")
    print("\nPer-category recall on held-out pr1m8 set:")
    print(cat_breakdown.to_string())

    cat_breakdown.to_csv(RESULTS / "cross_source_category_breakdown.csv")

    summary = pd.DataFrame([{
        "in_distribution_val_f1": best_f1,
        "ood_pr1m8_recall": ood_recall,
        "n_pr1m8": len(pr1m8),
    }])
    summary.to_csv(RESULTS / "cross_source_generalization_summary.csv", index=False)
    print(f"\nSaved to {RESULTS / 'cross_source_category_breakdown.csv'} and "
          f"{RESULTS / 'cross_source_generalization_summary.csv'}")


if __name__ == "__main__":
    main()
