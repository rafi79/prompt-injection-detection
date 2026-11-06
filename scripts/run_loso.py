"""
Leave-one-source-out (LOSO) generalisation for ALL systems.

For each seed and each source S in {jailbreak_llms, deepset, pr1m8}:
  train+validate on the other two sources (85/15 stratified split, threshold
  tuned on that validation split), test on S only.

pr1m8 note: 51 of its 82 rows are verbatim copies of deepset rows. Those rows
are attributed to deepset during cleaning, so the held-out pr1m8 test set
contains only the 31 rows that occur nowhere else. For transparency the
script ALSO scores the 51 duplicated rows (they are in the training data
whenever deepset is) and reports them separately as "pr1m8_seen".

Writes
  results/loso_per_seed.csv        metrics per seed / held-out source / system
  results/loso_summary.csv         mean +/- std (and 95% t-CI) across seeds
  results/loso_pr1m8_categories.csv per-category recall on pr1m8 (all systems)

    py scripts/run_loso.py                         # CPU systems, seeds 42 123 7
    py scripts/run_loso.py --systems B3            # add the GPU baseline
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_pool, make_split, fingerprint, SOURCES, SRC_PR, SRC_DS
from src.systems import make_system, CPU_SYSTEMS
from src.evalstats import metrics, wilson, t_ci95

RESULTS = ROOT / "results"


def upsert(path, df, keys):
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True).drop_duplicates(subset=keys, keep="last")
    df.to_csv(path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--systems", nargs="+", default=CPU_SYSTEMS)
    ap.add_argument("--heldout", nargs="+", default=SOURCES)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    for seed in args.seeds:
        pool = load_pool(seed)
        print(f"\n=== seed {seed}  data={fingerprint(pool)}", flush=True)
        for S in args.heldout:
            test = pool[pool.source == S].reset_index(drop=True)
            rest = pool[pool.source != S].reset_index(drop=True)
            train, val, _ = make_split(rest, seed, "random", fracs=(0.85, 0.15, 0.0))
            seen = pool[(pool.source == SRC_DS) & pool.pr1m8_category.notna()].reset_index(drop=True) \
                if S == SRC_PR else None
            print(f"  held-out {S}: test {len(test)} ({int(test.label.sum())} pos) | "
                  f"train {len(train)} ({int(train.label.sum())} pos) | val {len(val)}", flush=True)
            rows, cat_rows = [], []
            for name in args.systems:
                t0 = time.perf_counter()
                det = make_system(name, seed=seed)
                det.fit(train.text.tolist(), train.label.values, val.text.tolist(), val.label.values)
                sc = det.scores(test.text.tolist())
                pr = (sc >= det.threshold).astype(int)
                m = metrics(test.label.values, pr, sc)
                k, n = int(pr[test.label.values == 1].sum()), int(test.label.sum())
                lo, hi = wilson(k, n)
                m.update({"seed": seed, "heldout": S, "system": name, "threshold": det.threshold,
                          "val_f1": det.val_f1 if hasattr(det, "val_f1") else np.nan,
                          "detected": k, "recall_wilson_low": lo, "recall_wilson_high": hi})
                rows.append(m)
                msg = (f"    {name:6s} R={m['recall']:.3f} [{lo:.3f},{hi:.3f}] ({k}/{n})")
                if test.label.nunique() == 2:
                    msg += f"  P={m['precision']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.3f} AUC={m.get('roc_auc', np.nan):.3f}"
                if S == SRC_PR:
                    t2 = test.assign(pred=pr, subset="pr1m8_unique")
                    if seen is not None and len(seen):
                        ps = det.predict(seen.text.tolist())
                        ks = int(ps.sum())
                        rows.append({"seed": seed, "heldout": "pr1m8_seen(dup of deepset, in train)",
                                     "system": name, "recall": ks / len(seen), "detected": ks,
                                     "n": len(seen), "n_pos": len(seen)})
                        msg += f" | seen-dup rows R={ks / len(seen):.3f} ({ks}/{len(seen)})"
                        t2 = pd.concat([t2, seen.assign(pred=ps, subset="pr1m8_seen")])
                    for (sub, cat), g in t2.groupby(["subset", "pr1m8_category"]):
                        cat_rows.append({"seed": seed, "system": name, "subset": sub, "category": cat,
                                         "n": len(g), "detected": int(g.pred.sum())})
                print(msg + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
                del det
                if name.startswith("B3"):
                    import gc, torch
                    gc.collect()
                    torch.cuda.empty_cache()
            upsert(RESULTS / "loso_per_seed.csv", pd.DataFrame(rows), ["seed", "heldout", "system"])
            if cat_rows:
                upsert(RESULTS / "loso_pr1m8_categories.csv", pd.DataFrame(cat_rows),
                       ["seed", "system", "subset", "category"])

    per = pd.read_csv(RESULTS / "loso_per_seed.csv")
    summ = []
    for (S, name), g in per.groupby(["heldout", "system"], sort=False):
        r = {"heldout": S, "system": name, "n_seeds": len(g), "n_test": int(g.n.iloc[0])}
        for m in ["recall", "precision", "f1", "fpr", "roc_auc"]:
            if m in g and g[m].notna().any():
                v = g[m].dropna().values
                r[f"{m}_mean"], r[f"{m}_std"] = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
                r[f"{m}_ci95"] = t_ci95(v)
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "loso_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n##### LOSO summary (mean over seeds) #####")
    cols = [c for c in ["heldout", "system", "n_seeds", "n_test", "recall_mean", "recall_std", "precision_mean",
                        "f1_mean", "f1_std", "fpr_mean", "roc_auc_mean"] if c in summ]
    print(summ[cols].round(4).to_string(index=False))

    cpath = RESULTS / "loso_pr1m8_categories.csv"
    if cpath.exists():
        c = pd.read_csv(cpath)
        agg = c.groupby(["subset", "category", "system"]).agg(n=("n", "first"), detected=("detected", "sum"),
                                                             seeds=("seed", "nunique")).reset_index()
        agg["recall_mean"] = agg.detected / (agg.n * agg.seeds)
        agg.to_csv(RESULTS / "loso_pr1m8_categories_summary.csv", index=False)
        print("\n##### pr1m8 per-category recall (mean over seeds) #####")
        print(agg.pivot_table(index=["subset", "category", "n"], columns="system",
                              values="recall_mean").round(3).to_string())


if __name__ == "__main__":
    main()
