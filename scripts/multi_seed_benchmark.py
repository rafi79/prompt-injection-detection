"""
Multi-seed benchmark for statistical robustness.

Every prior result in this project was a single train/val/test split at a
fixed seed. This script repeats dataset construction + training + test-set
evaluation across 5 different seeds for every CPU system (B1, v1, v2, B2),
and reports mean +/- std and a 95% confidence interval per metric. GPU
systems (PromptGuard) are not included here - see the laptop-side scripts
for those, since this needs to run wherever the CPU-only pipeline lives.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sanitizer import sanitize
from src.keyword_engine import analyze
from src.risk_scorer import RiskScorer, extract_features
from src.semantic_lite import SemanticLiteEncoder
from src.baselines import KeywordOnlyBaseline, TfidfSvmBaseline

RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

SEEDS = [42, 123, 7, 2024, 99]


def load_jailbreak_llms():
    jb = pd.read_csv(RAW / "jailbreak_llms_prompts" / "jailbreak_prompts_2023_12_25.csv")
    reg = pd.read_csv(RAW / "jailbreak_llms_prompts" / "regular_prompts_2023_12_25.csv")
    return (pd.DataFrame({"text": jb["prompt"].astype(str), "label": 1}),
            pd.DataFrame({"text": reg["prompt"].astype(str), "label": 0}))


def load_deepset():
    tr = pd.read_parquet(RAW / "deepset_train.parquet")
    te = pd.read_parquet(RAW / "deepset_test.parquet")
    df = pd.concat([tr, te], ignore_index=True)
    return df[df.label == 1][["text", "label"]], df[df.label == 0][["text", "label"]]


def load_pr1m8():
    df = pd.read_csv(RAW / "pr1m8_prompt_injections.csv")
    return pd.DataFrame({"text": df["text"].astype(str), "label": 1}), pd.DataFrame(columns=["text", "label"])


def build_split(seed):
    jb_pos, jb_neg = load_jailbreak_llms()
    ds_pos, ds_neg = load_deepset()
    pr_pos, pr_neg = load_pr1m8()

    positives = pd.concat([jb_pos, ds_pos, pr_pos], ignore_index=True)
    negatives = pd.concat([jb_neg, ds_neg, pr_neg], ignore_index=True)
    for df in (positives, negatives):
        df["text"] = df["text"].astype(str).str.strip()
    positives = positives[positives.text.str.len() > 5].drop_duplicates(subset="text")
    negatives = negatives[negatives.text.str.len() > 5].drop_duplicates(subset="text")

    n_pos = len(positives)
    negatives = negatives.sample(n=min(len(negatives), n_pos * 3), random_state=seed)

    pos = positives.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    neg = negatives.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def split(df):
        n = len(df)
        n_train, n_val = int(n * 0.70), int(n * 0.15)
        return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]

    pos_tr, pos_va, pos_te = split(pos)
    neg_tr, neg_va, neg_te = split(neg)
    train = pd.concat([pos_tr, neg_tr]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat([pos_va, neg_va]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat([pos_te, neg_te]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for df in (train, val, test):
        df["label"] = df["label"].astype(int)
    return train, val, test


def build_v1_matrix(texts):
    feats = [extract_features(sanitize(t), analyze(sanitize(t)), source_trust=0.0) for t in texts]
    return np.vstack(feats)


def build_v2_matrix(texts, encoder):
    clean_texts = [sanitize(t) for t in texts]
    sem_feats = encoder.similarity_features(clean_texts)
    feats = [extract_features(c, analyze(c), source_trust=0.0, semantic_feats=s)
             for c, s in zip(clean_texts, sem_feats)]
    return np.vstack(feats)


def tune_threshold(scorer, X_val, y_val):
    probs = scorer.predict_proba(X_val)
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    scorer.flag_threshold = float(best_t)


def run_one_seed(seed):
    train, val, test = build_split(seed)
    y_train, y_val, y_test = train.label.values, val.label.values, test.label.values
    row = {"seed": seed}

    # --- B1: keyword-only ---
    t0 = time.perf_counter()
    kw = KeywordOnlyBaseline(hit_threshold=1)
    preds_b1, lats_b1 = [], []
    for t in test.text.tolist():
        t1 = time.perf_counter()
        preds_b1.append(kw.predict_label(t))
        lats_b1.append((time.perf_counter() - t1) * 1000.0)
    row["B1_accuracy"] = accuracy_score(y_test, preds_b1)
    row["B1_precision"] = precision_score(y_test, preds_b1, zero_division=0)
    row["B1_recall"] = recall_score(y_test, preds_b1, zero_division=0)
    row["B1_f1"] = f1_score(y_test, preds_b1, zero_division=0)
    row["B1_latency_ms"] = float(np.mean(lats_b1))

    # --- v1: lexical risk scorer ---
    X_train_v1, X_val_v1, X_test_v1 = build_v1_matrix(train.text.tolist()), build_v1_matrix(val.text.tolist()), build_v1_matrix(test.text.tolist())
    scorer_v1 = RiskScorer()
    scorer_v1.fit(X_train_v1, y_train)
    tune_threshold(scorer_v1, X_val_v1, y_val)
    probs_v1 = scorer_v1.predict_proba(X_test_v1)
    preds_v1 = (probs_v1 >= scorer_v1.flag_threshold).astype(int)
    row["v1_accuracy"] = accuracy_score(y_test, preds_v1)
    row["v1_precision"] = precision_score(y_test, preds_v1, zero_division=0)
    row["v1_recall"] = recall_score(y_test, preds_v1, zero_division=0)
    row["v1_f1"] = f1_score(y_test, preds_v1, zero_division=0)

    # --- v2: lexical + semantic-lite ---
    encoder = SemanticLiteEncoder(n_components=100)
    encoder.fit(train.text.tolist(), y_train)
    X_train_v2, X_val_v2, X_test_v2 = (build_v2_matrix(train.text.tolist(), encoder),
                                        build_v2_matrix(val.text.tolist(), encoder),
                                        build_v2_matrix(test.text.tolist(), encoder))
    scorer_v2 = RiskScorer()
    scorer_v2.fit(X_train_v2, y_train)
    tune_threshold(scorer_v2, X_val_v2, y_val)
    probs_v2 = scorer_v2.predict_proba(X_test_v2)
    preds_v2 = (probs_v2 >= scorer_v2.flag_threshold).astype(int)
    row["v2_accuracy"] = accuracy_score(y_test, preds_v2)
    row["v2_precision"] = precision_score(y_test, preds_v2, zero_division=0)
    row["v2_recall"] = recall_score(y_test, preds_v2, zero_division=0)
    row["v2_f1"] = f1_score(y_test, preds_v2, zero_division=0)

    # --- B2: TF-IDF + SVM ---
    svm = TfidfSvmBaseline()
    svm.fit(train.text.tolist(), train.label.tolist())
    preds_b2 = [svm.predict_label(t) for t in test.text.tolist()]
    row["B2_accuracy"] = accuracy_score(y_test, preds_b2)
    row["B2_precision"] = precision_score(y_test, preds_b2, zero_division=0)
    row["B2_recall"] = recall_score(y_test, preds_b2, zero_division=0)
    row["B2_f1"] = f1_score(y_test, preds_b2, zero_division=0)

    return row


def ci95(values):
    values = np.array(values)
    n = len(values)
    if n < 2:
        return 0.0
    sem = stats.sem(values)
    return sem * stats.t.ppf(0.975, n - 1)


def main():
    rows = []
    for seed in SEEDS:
        print(f"=== Seed {seed} ===")
        row = run_one_seed(seed)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "multi_seed_raw.csv", index=False)

    systems = ["B1", "v1", "v2", "B2"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    summary_rows = []
    for sys_name in systems:
        row = {"system": sys_name}
        for m in metrics:
            col = f"{sys_name}_{m}"
            vals = df[col].values
            row[f"{m}_mean"] = np.mean(vals)
            row[f"{m}_std"] = np.std(vals, ddof=1)
            row[f"{m}_ci95"] = ci95(vals)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 200)
    print("\n=== Summary across", len(SEEDS), "seeds (mean +/- std, 95% CI) ===")
    for _, r in summary.iterrows():
        print(f"\n{r['system']}:")
        for m in metrics:
            print(f"  {m}: {r[f'{m}_mean']:.4f} +/- {r[f'{m}_std']:.4f}  (95% CI +/- {r[f'{m}_ci95']:.4f})")

    summary.to_csv(RESULTS / "multi_seed_summary.csv", index=False)
    print(f"\nSaved raw per-seed results to {RESULTS / 'multi_seed_raw.csv'}")
    print(f"Saved summary to {RESULTS / 'multi_seed_summary.csv'}")


if __name__ == "__main__":
    main()
