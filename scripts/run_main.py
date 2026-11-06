"""
Main multi-seed evaluation (replaces benchmark.py, semantic_lite_experiment.py
and multi_seed_benchmark.py; all systems now share one data version, one
training protocol and one thresholding rule).

For every seed and split mode:
  build the pinned dataset -> split -> train every system -> test.
Writes
  results/main_<mode>_per_seed.csv       metrics per seed and system
  results/main_<mode>_summary.csv        mean, std, 95% t-CI across seeds
  results/main_<mode>_paired_tests.csv   McNemar + paired bootstrap per seed,
                                         paired t-test across seeds
  results/main_<mode>_length_matched.csv metrics on a length-matched,
                                         class-balanced test subset
  results/preds/main_<mode>_s<seed>.csv  per-example scores/predictions
  results/main_dataset_report.json       cleaning report + data fingerprints
and, for seed 42 / random split, the trained models used by the other
scripts (results/models/).

Examples (PowerShell):
  py scripts/run_main.py                                   # CPU systems, 5 seeds, random + grouped
  py scripts/run_main.py --systems B3 B3raw --modes random # GPU baseline, appended to the same CSVs
"""
import sys
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_pool, make_split, near_duplicate_groups, fingerprint, describe, norm_key
from src.systems import make_system, CPU_SYSTEMS, DIAGNOSTIC_SYSTEMS
from src.evalstats import metrics, t_ci95, mcnemar_exact, paired_bootstrap_f1, paired_t

RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
MODELS = RESULTS / "models"
SEEDS = [42, 123, 7, 2024, 99]
PAIRS = [("v2", "v1"), ("v2", "B2"), ("v2cf", "v2"), ("v2cf", "B2"), ("v3", "v2cf"), ("v3", "B2"),
         ("B3", "B2"), ("B3", "v3"), ("B3", "v2")]


def length_matched_subset(test, seed, n_bins=10):
    rng = np.random.default_rng(seed)
    loglen = np.log1p(test.text.str.len().values)
    edges = np.unique(np.quantile(loglen, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, loglen, side="right") - 1, 0, len(edges) - 2)
    keep = []
    for k in np.unique(b):
        pos = np.where((b == k) & (test.label.values == 1))[0]
        neg = np.where((b == k) & (test.label.values == 0))[0]
        m = min(len(pos), len(neg))
        if m:
            keep += list(rng.choice(pos, m, replace=False)) + list(rng.choice(neg, m, replace=False))
    return np.array(sorted(keep))


def upsert(path, df, keys):
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, df], ignore_index=True)
        df = merged.drop_duplicates(subset=keys, keep="last")
    df.to_csv(path, index=False)
    return df


def summarise(per_seed, mode):
    rows = []
    for sys_name, g in per_seed.groupby("system", sort=False):
        r = {"system": sys_name, "n_seeds": len(g)}
        for m in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "val_f1", "threshold"]:
            if m in g:
                v = g[m].astype(float).values
                r[f"{m}_mean"], r[f"{m}_std"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                r[f"{m}_ci95"] = t_ci95(v)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"main_{mode}_summary.csv", index=False)
    return out


def paired_tests(mode, seeds):
    rows = []
    per_seed_f1 = {}
    for s in seeds:
        p = PREDS / f"main_{mode}_s{s}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for a, b in PAIRS:
            if f"{a}_pred" not in d or f"{b}_pred" not in d:
                continue
            nb, nc, pm = mcnemar_exact(d.label, d[f"{a}_pred"], d[f"{b}_pred"])
            obs, lo, hi, p1 = paired_bootstrap_f1(d.label.values, d[f"{a}_pred"].values, d[f"{b}_pred"].values,
                                                 seed=s)
            rows.append({"seed": s, "a": a, "b": b, "a_right_b_wrong": nb, "a_wrong_b_right": nc,
                         "mcnemar_p": pm, "f1_diff": obs, "boot_ci_low": lo, "boot_ci_high": hi,
                         "boot_p_diff_le_0": p1})
    df = pd.DataFrame(rows)
    if len(df):
        ps = pd.read_csv(RESULTS / f"main_{mode}_per_seed.csv")
        agg = []
        for (a, b), _ in df.groupby(["a", "b"], sort=False):
            fa = ps[ps.system == a].set_index("seed").f1
            fb = ps[ps.system == b].set_index("seed").f1
            common = sorted(set(fa.index) & set(fb.index))
            md, pt = paired_t(fa.loc[common].values, fb.loc[common].values)
            agg.append({"seed": "all(paired t)", "a": a, "b": b, "f1_diff": md, "mcnemar_p": np.nan,
                        "paired_t_p": pt, "n_seeds": len(common)})
        df = pd.concat([df, pd.DataFrame(agg)], ignore_index=True)
    df.to_csv(RESULTS / f"main_{mode}_paired_tests.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--systems", nargs="+", default=CPU_SYSTEMS + DIAGNOSTIC_SYSTEMS)
    ap.add_argument("--modes", nargs="+", default=["random", "grouped"], choices=["random", "grouped"])
    args = ap.parse_args()
    for d in (RESULTS, PREDS, MODELS):
        d.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS / "main_dataset_report.json"
    report_all = json.loads(report_path.read_text()) if report_path.exists() else {}

    for mode in args.modes:
        for seed in args.seeds:
            rep = {}
            pool = load_pool(seed, report=rep)
            groups = near_duplicate_groups(pool.text.tolist()) if mode == "grouped" else None
            train, val, test = make_split(pool, seed, mode, groups=groups)
            rep.update({"fingerprint_pool": fingerprint(pool), "fingerprint_test": fingerprint(test),
                        "split": describe(train, val, test)})
            if groups is not None:
                rep["near_dup_groups"] = int(len(set(groups)))
                rep["texts_in_multi_member_groups"] = int(pd.Series(groups).map(
                    pd.Series(groups).value_counts()).gt(1).sum())
            report_all[f"{mode}_s{seed}"] = {k: (str(v) if not isinstance(v, (int, float, str)) else v)
                                            for k, v in rep.items()}
            report_path.write_text(json.dumps(report_all, indent=2))
            print(f"\n=== mode={mode} seed={seed}  data={rep['fingerprint_pool']}  {rep['split']}", flush=True)

            pred_path = PREDS / f"main_{mode}_s{seed}.csv"
            pred_df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame({
                "key": test.text.map(lambda t: norm_key(t)[:80]), "label": test.label.values,
                "source": test.source.values, "length": test.text.str.len().values})
            lm_idx = length_matched_subset(test, seed)

            rows, lm_rows = [], []
            for name in args.systems:
                t0 = time.perf_counter()
                det = make_system(name, seed=seed)
                det.fit(train.text.tolist(), train.label.values, val.text.tolist(), val.label.values)
                fit_s = time.perf_counter() - t0
                sc = det.scores(test.text.tolist())
                pr = (sc >= det.threshold).astype(int)
                m = metrics(test.label.values, pr, sc)
                m.update({"system": name, "seed": seed, "mode": mode, "threshold": det.threshold,
                          "val_f1": getattr(det, "val_f1", np.nan), "fit_seconds": fit_s,
                          "data_fingerprint": rep["fingerprint_pool"]})
                rows.append(m)
                pred_df[f"{name}_score"], pred_df[f"{name}_pred"] = sc, pr
                lm = metrics(test.label.values[lm_idx], pr[lm_idx], sc[lm_idx])
                lm.update({"system": name, "seed": seed, "mode": mode})
                lm_rows.append(lm)
                print(f"  {name:8s} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                      f"AUC={m.get('roc_auc', float('nan')):.4f} FPR={m['fpr']:.4f} thr={det.threshold:.2f} "
                      f"| length-matched AUC={lm.get('roc_auc', float('nan')):.4f} "
                      f"(n={lm['n']})  fit={fit_s:.1f}s", flush=True)
                if mode == "random" and seed == 42:
                    if name in ("B3",):
                        det.save(str(MODELS / "b3_seed42"))
                    elif name not in ("B3raw",) and ":" not in name:  # ablation variants are not reused (":" is invalid in Windows file names)
                        with open(MODELS / f"{name}_seed42.pkl", "wb") as f:
                            pickle.dump(det, f)
                del det
                import gc
                gc.collect()
                if name.startswith("B3"):
                    import torch
                    torch.cuda.empty_cache()

            pred_df.to_csv(pred_path, index=False)
            per_seed = upsert(RESULTS / f"main_{mode}_per_seed.csv", pd.DataFrame(rows), ["system", "seed"])
            upsert(RESULTS / f"main_{mode}_length_matched.csv", pd.DataFrame(lm_rows), ["system", "seed"])

        per_seed = pd.read_csv(RESULTS / f"main_{mode}_per_seed.csv")
        summary = summarise(per_seed, mode)
        lm = pd.read_csv(RESULTS / f"main_{mode}_length_matched.csv")
        lm_sum = lm.groupby("system", sort=False)[["roc_auc", "recall", "fpr", "f1"]].agg(["mean", "std"])
        lm_sum.to_csv(RESULTS / f"main_{mode}_length_matched_summary.csv")
        tests = paired_tests(mode, sorted(set(per_seed.seed)))

        pd.set_option("display.width", 220)
        print(f"\n##### SUMMARY mode={mode} (mean +/- std, 95% CI) #####")
        for _, r in summary.iterrows():
            print(f"  {r.system:8s} n={int(r.n_seeds)}  F1 {r.f1_mean:.4f} +/- {r.f1_std:.4f} (CI +/-{r.f1_ci95:.4f})"
                  f"  P {r.precision_mean:.4f}  R {r.recall_mean:.4f}  FPR {r.fpr_mean:.4f}"
                  f"  AUC {r.get('roc_auc_mean', float('nan')):.4f}")
        print(f"\n##### length-matched test subset (balanced), mode={mode} #####")
        print(lm_sum.round(4).to_string())
        agg = tests[tests.seed.astype(str) == "all(paired t)"] if len(tests) else tests
        if len(agg):
            print(f"\n##### paired tests across seeds, mode={mode} #####")
            print(agg[["a", "b", "f1_diff", "paired_t_p", "n_seeds"]].round(4).to_string(index=False))
            per = tests[tests.seed.astype(str) != "all(paired t)"]
            print("\nper-seed McNemar p-values:")
            print(per.pivot_table(index=["a", "b"], columns="seed", values="mcnemar_p").round(4).to_string())


if __name__ == "__main__":
    main()
