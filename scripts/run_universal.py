"""
Universal evasion suffix against the DETECTORS (reviewer R6-e, "universal").

Gradient-based universal prompt-injection attacks (Liu et al., 2024) optimise
strings that make the downstream LLM comply; they do not target a separate
detector, so we do not re-run them. The detector-side analogue we CAN test
honestly: learn ONE short word suffix that, appended to any attack prompt,
pushes the detector's score below its threshold.

Black-box greedy coordinate search (score access only):
  vocabulary  = the 60 most frequent content words of BENIGN training prompts
                (data-driven; no access to model weights)
  for each of L=8 positions: try every vocabulary word, keep the one that
                minimises the mean risk score over 150 TRAINING attack prompts
  evaluate    on the 150 held-out TEST attack prompts (same sample as the
                other attack scripts): evasion rate with the learned suffix,
                vs. no suffix and vs. 5 random suffixes of the same length
  transfer    suffix learned on system A applied to every system B

The attack text itself is unchanged (only benign words are appended), so the
malicious intent is preserved by construction.

Writes results/universal_suffix.csv, results/universal_transfer.csv

    py scripts/run_universal.py --with-b3
"""
import re
import sys
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.attacks import attack_sample
from src.data import load_pool, make_split
from src.systems import WindowedDetector
from src.trained import load_many

RESULTS = ROOT / "results"
STOP = set("""the a an and or of to in on for with is are be this that it as at by from your you
your i me my we our will can please about into what which who how when where than then them they
their there have has had was were been not but if so do does did just also more most very any all
each other some such only own same too should would could must may might shall""".split())


def vocabulary(k=60):
    pool = load_pool(42)
    tr, _, _ = make_split(pool, 42, "random")
    neg = Counter(w for t in tr[tr.label == 0].text for w in re.findall(r"[a-z]{4,12}", t.lower()))
    pos = Counter(w for t in tr[tr.label == 1].text for w in re.findall(r"[a-z]{4,12}", t.lower()))
    n_neg, n_pos = max(sum(neg.values()), 1), max(sum(pos.values()), 1)
    cands = [w for w, c in neg.most_common(2000)
             if w not in STOP and c / n_neg > 1.5 * pos.get(w, 0) / n_pos]
    return cands[:k]


def apply(prompts, suffix):
    return [p + " " + " ".join(suffix) if suffix else p for p in prompts]


def learn_suffix(det, train_prompts, vocab, length):
    suffix = []
    for pos in range(length):
        best_w, best = None, np.inf
        for w in vocab:
            s = det.scores(apply(train_prompts, suffix + [w])).mean()
            if s < best:
                best, best_w = s, w
        suffix.append(best_w)
        ev = (det.scores(apply(train_prompts, suffix)) < det.threshold).mean()
        print(f"    pos {pos + 1}: +'{best_w}'  mean score {best:.4f}  train evasion {ev:.3f}", flush=True)
    return suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["B1", "v1", "v2", "v2cf", "v3", "B2"])
    ap.add_argument("--with-b3", action="store_true")
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=60)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--window", action="store_true", help="also attack windowed versions of v2cf/v3/B2")
    args = ap.parse_args()

    train_prompts = attack_sample(args.n, split="train")
    test_prompts = attack_sample(args.n, split="test")
    vocab = vocabulary(args.vocab)
    print(f"vocabulary ({len(vocab)}): {' '.join(vocab)}")
    dets = load_many(args.systems + (["B3"] if args.with_b3 else []))
    if args.window:
        pool = load_pool(42)
        _, val, _ = make_split(pool, 42, "random")
        for b in ("v2cf", "v3", "B2"):
            if b in dets:
                w = WindowedDetector(dets[b]).fit(None, None, val.text.tolist(), val.label.values)
                dets[w.name] = w

    rng = np.random.default_rng(0)
    rows, suffixes = [], {}
    for name, det in dets.items():
        print(f"\n=== {name} ===", flush=True)
        if name == "B1":
            print("  keyword counts cannot be lowered by appending benign words; suffix search skipped")
        base_flag = det.predict(test_prompts)
        flagged = [p for p, f in zip(test_prompts, base_flag) if f]
        # optimise only over the training attacks this detector flags
        tr_flagged = [p for p, f in zip(train_prompts, det.predict(train_prompts)) if f]
        suffix = learn_suffix(det, tr_flagged, vocab, args.length) if (name != "B1" and tr_flagged) else []
        suffixes[name] = suffix
        ev_learned = float((det.scores(apply(flagged, suffix)) < det.threshold).mean()) if flagged else np.nan
        ev_rand = []
        for _ in range(5):
            rs = list(rng.choice(vocab, args.length, replace=True))
            ev_rand.append(float((det.scores(apply(flagged, rs)) < det.threshold).mean()) if flagged else np.nan)
        rows.append({"system": name, "suffix": " ".join(suffix), "n_test_flagged": len(flagged),
                     "n_test": len(test_prompts), "n_train_flagged": len(tr_flagged), "evasion_learned_suffix": ev_learned,
                     "evasion_random_suffix_mean": float(np.mean(ev_rand)),
                     "evasion_random_suffix_std": float(np.std(ev_rand, ddof=1))})
        print(f"  held-out evasion: learned {ev_learned:.3f} | random {np.mean(ev_rand):.3f} "
              f"(+/- {np.std(ev_rand, ddof=1):.3f}) | flagged originally {len(flagged)}/{len(test_prompts)}")

    pd.DataFrame(rows).to_csv(RESULTS / "universal_suffix.csv", index=False)
    tr = []
    for a, suf in suffixes.items():
        for b, det in dets.items():
            flagged = [p for p, f in zip(test_prompts, det.predict(test_prompts)) if f]
            ev = float((det.scores(apply(flagged, suf)) < det.threshold).mean()) if (flagged and suf) else np.nan
            tr.append({"suffix_from": a, "evaluated_on": b, "evasion": ev})
    t = pd.DataFrame(tr)
    t.to_csv(RESULTS / "universal_transfer.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    print("\ntransfer (rows: suffix learned on; cols: evaluated on)")
    print(t.pivot(index="suffix_from", columns="evaluated_on", values="evasion").round(3).to_string())


if __name__ == "__main__":
    main()
