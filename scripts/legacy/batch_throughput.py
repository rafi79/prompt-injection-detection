"""
Batch throughput test.

Every latency number reported so far measured single-request (unbatched)
inference - the realistic scenario for a real-time chat assistant handling
one incoming prompt at a time. But heavier models are often deployed with
batching in production (e.g. bulk content moderation, offline scanning).
This script measures throughput (texts/sec) at increasing batch sizes for
every system, to show WHERE each one's advantage actually holds:
  - The lightweight CPU pipeline (v1/v2) should be roughly batch-size
    invariant, since its cost is dominated by per-example Python/regex
    work rather than a model forward pass.
  - TF-IDF+SVM should benefit moderately from batching (vectorized
    sklearn operations).
  - The fine-tuned PromptGuard transformer should benefit dramatically
    from batching (GPU parallelism), potentially closing or reversing the
    single-request latency gap at large batch sizes.

Requires: results/risk_scorer.pkl, results/risk_scorer_v2.pkl,
results/tfidf_svm.pkl (from earlier scripts) and
results/promptguard_finetuned_model/ (from scripts/promptguard_finetune.py).
"""
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.sanitizer import sanitize
from src.keyword_engine import analyze
from src.risk_scorer import extract_features

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

BATCH_SIZES = [1, 8, 16, 32, 64, 128]
N_REPEATS = 3  # repeat each batch-size measurement to reduce noise


def bench(name, run_batch_fn, texts, batch_sizes=BATCH_SIZES, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for bs in batch_sizes:
        # sample once per batch size (not a deterministic prefix - avoids
        # confounding batch size with systematic text-length differences),
        # then time that same fixed sample repeatedly for a stable min-of-N
        idx = rng.choice(len(texts), size=bs, replace=(bs > len(texts)))
        batch = [texts[i] for i in idx]
        avg_len = sum(len(t) for t in batch) / len(batch)

        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            run_batch_fn(batch)
            times.append(time.perf_counter() - t0)
        best_time = min(times)  # best-of-N, standard practice to reduce OS noise
        throughput = bs / best_time
        rows.append({"system": name, "batch_size": bs, "throughput_req_s": throughput,
                      "batch_latency_ms": best_time * 1000.0, "per_item_ms": best_time * 1000.0 / bs,
                      "batch_avg_char_len": avg_len})
        print(f"  {name}: batch={bs:4d}  throughput={throughput:8.1f} req/s  "
              f"per-item={best_time*1000.0/bs:.3f} ms  (avg_len={avg_len:.0f} chars)")
    return rows


def main():
    test = pd.read_csv(DATA / "test.csv")
    texts = test.text.tolist()

    all_rows = []

    # --- v1: lexical risk scorer (loop-based, expect ~flat throughput) ---
    with open(RESULTS / "risk_scorer.pkl", "rb") as f:
        scorer_v1 = pickle.load(f)

    def run_v1(batch):
        for t in batch:
            clean = sanitize(t)
            kw = analyze(clean)
            feats = extract_features(clean, kw, source_trust=0.0)
            scorer_v1.predict_proba(feats.reshape(1, -1))

    print("\n--- v1: Lexical risk scorer ---")
    all_rows += bench("v1: Lexical", run_v1, texts)

    # --- v2: lexical + semantic-lite (partially vectorized: TF-IDF/SVD batch) ---
    with open(RESULTS / "risk_scorer_v2.pkl", "rb") as f:
        v2 = pickle.load(f)
    scorer_v2, encoder = v2["scorer"], v2["encoder"]

    def run_v2(batch):
        clean_texts = [sanitize(t) for t in batch]
        sem_feats = encoder.similarity_features(clean_texts)  # vectorized over the batch
        feat_matrix = np.vstack([
            extract_features(c, analyze(c), source_trust=0.0, semantic_feats=s)
            for c, s in zip(clean_texts, sem_feats)
        ])
        scorer_v2.predict_proba(feat_matrix)

    print("\n--- v2: Lexical + Semantic-lite ---")
    all_rows += bench("v2: +Semantic-lite", run_v2, texts)

    # --- B2: TF-IDF + SVM (fully vectorized sklearn batch call) ---
    with open(RESULTS / "tfidf_svm.pkl", "rb") as f:
        tfidf_svm = pickle.load(f)

    def run_svm(batch):
        clean_texts = [sanitize(t) for t in batch]
        X = tfidf_svm.vectorizer.transform(clean_texts)
        tfidf_svm.clf.predict(X)

    print("\n--- B2: TF-IDF + Linear SVM ---")
    all_rows += bench("B2: TF-IDF+SVM", run_svm, texts)

    # --- B3: fine-tuned PromptGuard (true GPU batched inference) ---
    model_dir = RESULTS / "promptguard_finetuned_model"
    if model_dir.exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n--- B3: PromptGuard-86M fine-tuned ({device}) ---")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()

        @torch.no_grad()
        def run_promptguard(batch):
            enc = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
            model(**enc).logits
            if device == "cuda":
                torch.cuda.synchronize()

        all_rows += bench("B3: PromptGuard-86M (GPU)", run_promptguard, texts,
                           batch_sizes=[1, 8, 16, 32, 64])
    else:
        print(f"\nSkipping B3 (no fine-tuned model found at {model_dir}; "
              f"run scripts/promptguard_finetune.py first)")

    df = pd.DataFrame(all_rows)
    pd.set_option("display.width", 140)
    print("\n" + df.to_string(index=False))
    df.to_csv(RESULTS / "batch_throughput_results.csv", index=False)
    print(f"\nSaved to {RESULTS / 'batch_throughput_results.csv'}")

    # crossover summary: at what batch size does each heavier system's
    # per-item latency drop below the lightweight system's single-request latency?
    v1_single = df[(df.system == "v1: Lexical") & (df.batch_size == 1)].per_item_ms.values[0]
    print(f"\nv1 single-request per-item latency: {v1_single:.3f} ms")
    for name in df.system.unique():
        if name == "v1: Lexical":
            continue
        sub = df[df.system == name]
        below = sub[sub.per_item_ms <= v1_single]
        if len(below):
            print(f"  {name} matches/beats v1's per-item latency starting at batch_size={below.batch_size.min()}")
        else:
            print(f"  {name} never beats v1's per-item latency within tested batch sizes")


if __name__ == "__main__":
    main()
