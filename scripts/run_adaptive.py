"""
Adaptive black-box attack with a query budget (reviewer R6-e).

Threat model: the attacker can submit a text to the detector and observe its
risk SCORE (soft-label black box; a strong, upper-bound attacker). Starting
from an attack prompt the detector flags, the attacker iteratively mutates
it; every detector call counts as one query. Success = score below the
detector's threshold. We report attack success rate (ASR) at budgets
1, 5, 10, 20 queries.

Mutation operators (src/attacks.py):
  synonym        swap an attack-salient word using a small hand-written list
  benign_padding prepend/append a benign sentence from the training data
                 (feature-space "dilution" of a document-level score)
  char           zero-width split / homoglyph / adjacent-letter typo in a word
  paraphrase     replace with an LLM paraphrase of the original (from
                 results/paraphrase_generations.csv, auto-valid ones only)
Search: each step proposes 4 mutations of the current best text (4 queries)
and keeps the lowest-scoring one (greedy, score-guided).

Mitigation: the same attack against windowed max-pooling versions
("+win") of v2cf, v3 and B2, and their clean test F1/FPR cost.

Prompts: the same 150 test attack prompts as run_paraphrase.py.
Writes results/adaptive_asr.csv, results/adaptive_examples.csv,
results/adaptive_window_cost.csv and results/adaptive_validity_sheet.csv
(50 successful adversarial texts to annotate: intent_preserved 1/0).

    py scripts/run_adaptive.py --with-b3
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.attacks import attack_sample, benign_sentences, OPERATORS
from src.data import load_pool, make_split
from src.systems import WindowedDetector
from src.trained import load_many
from src.evalstats import metrics

RESULTS = ROOT / "results"
BUDGETS = [1, 5, 10, 20]


def attack_one(det, text, rng, budget, ops, benign_pool, paraphrases):
    used = set()
    cur, cur_score = text, det.score_one(text)
    q = 0
    trace = []
    while q < budget:
        props = []
        for _ in range(4):
            if q >= budget:
                break
            name = ops[rng.integers(len(ops))]
            new = None
            for _try in range(5):
                new = OPERATORS[name](cur if name != "paraphrase" else text, rng, benign_pool=benign_pool,
                                      paraphrases=paraphrases, used=used)
                if new is not None:
                    break
                name = ops[rng.integers(len(ops))]
            if new is None:
                continue
            s = det.score_one(new)
            q += 1
            props.append((s, new, name))
            trace.append(name)
            if s < det.threshold:
                return q, new, trace
        if not props:
            break
        best = min(props, key=lambda x: x[0])
        if best[0] < cur_score:
            cur_score, cur = best[0], best[1]
    return None, cur, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["B1", "v1", "v2", "v2cf", "v3", "B2"])
    ap.add_argument("--with-b3", action="store_true")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--budget", type=int, default=max(BUDGETS))
    ap.add_argument("--operators", nargs="+", default=list(OPERATORS))
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    prompts = attack_sample(args.n)
    benign_pool = benign_sentences()
    para = {}
    gen = RESULTS / "paraphrase_generations.csv"
    ops = list(args.operators)
    if "paraphrase" in ops:
        if gen.exists():
            g = pd.read_csv(gen).fillna("")
            g = g[g.auto_invalid == ""]
            for _, r in g.iterrows():
                para.setdefault(r.original, []).append(r.paraphrase)
            print(f"LLM paraphrase pool: {sum(map(len, para.values()))} texts for {len(para)} prompts")
        else:
            print("no paraphrase_generations.csv -> running WITHOUT the paraphrase operator")
            ops.remove("paraphrase")

    dets = load_many(args.systems + (["B3"] if args.with_b3 else []))
    cost_rows = []
    if not args.no_window:
        pool = load_pool(42)
        _, val, test = make_split(pool, 42, "random")
        for base in ("v2cf", "v3", "B2"):
            if base in dets:
                w = WindowedDetector(dets[base]).fit(None, None, val.text.tolist(), val.label.values)
                for d in (dets[base], w):
                    sc = d.scores(test.text.tolist())
                    m = metrics(test.label.values, (sc >= d.threshold).astype(int), sc)
                    m["system"] = d.name
                    cost_rows.append(m)
                    print(f"  clean test {d.name:9s} F1={m['f1']:.4f} FPR={m['fpr']:.4f} R={m['recall']:.4f}")
                dets[w.name] = w
        pd.DataFrame(cost_rows).to_csv(RESULTS / "adaptive_window_cost.csv", index=False)

    rows, examples = [], []
    for name, det in dets.items():
        rng = np.random.default_rng(0)
        flagged = det.predict(prompts)
        succ_q = []
        for text, f in zip(prompts, flagged):
            if not f:
                continue
            q, adv, trace = attack_one(det, text, rng, args.budget, ops, benign_pool, para.get(text, []))
            succ_q.append(q)
            if q is not None:
                examples.append({"system": name, "queries": q, "original": text, "adversarial": adv,
                                 "operators_tried": "|".join(trace),
                                 "final_operator": trace[-1]})
        n_att = len(succ_q)
        for b in BUDGETS:
            k = sum(1 for q in succ_q if q is not None and q <= b)
            rows.append({"system": name, "budget": b, "n_flagged_originally": n_att,
                         "n_not_flagged_originally": int(len(prompts) - n_att),
                         "successes": k, "asr": k / max(n_att, 1)})
        print(f"{name:9s} flagged {n_att}/{len(prompts)} | ASR " +
              "  ".join(f"@{b}={r['asr']:.3f}" for b, r in zip(BUDGETS, rows[-len(BUDGETS):])), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "adaptive_asr.csv", index=False)
    ex = pd.DataFrame(examples)
    ex.to_csv(RESULTS / "adaptive_examples.csv", index=False)
    if len(ex):
        sheet = ex.sample(n=min(50, len(ex)), random_state=0)[["system", "original", "adversarial", "final_operator"]]
        sheet["intent_preserved"] = ""
        sheet.to_csv(RESULTS / "adaptive_validity_sheet.csv", index=False)
        print("operator that produced the evading text:", ex.final_operator.value_counts().to_dict())
    print("\n" + df.pivot(index="system", columns="budget", values="asr").round(3).to_string())


if __name__ == "__main__":
    main()
