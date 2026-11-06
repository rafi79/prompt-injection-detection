"""
Latency-aware cascade experiments (stage 1 = B2 or v3 on CPU, stage 2 =
fine-tuned PromptGuard B3 only for uncertain inputs).

Part "offline" (no GPU, uses the saved per-example test predictions of
run_main.py): all seeds. Because validation scores were not saved, the band
and all thresholds are chosen by 2-fold cross-fitting on the test
predictions (choose on one half, apply to the other, swap), applied
identically to every compared system. Paired McNemar per seed and paired
t-test across seeds.

Part "live" (GPU, seed 42 models from results/models/): the band is chosen
on the real validation split, then every cascade is measured with the same
protocols as the single systems: test accuracy, single-request latency,
obfuscation suite + benign controls, paraphrase evasion, adaptive
black-box attack (query budget 20) and multilingual recall/FPR.

    py scripts/run_cascade.py offline
    py scripts/run_cascade.py live
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.cascade import CascadeDetector, choose_band, cascade_predict
from src.evalstats import metrics, mcnemar_exact, paired_t, t_ci95
from src.systems import Detector, tune_threshold

RESULTS = ROOT / "results"
BUDGETS = [0.05, 0.10, 0.20, 0.30]


# ======================================================================
# offline: all seeds, from saved test predictions
# ======================================================================
def crossfit_single(score, y, halves):
    pred = np.zeros(len(y), dtype=int)
    for k in range(2):
        tu, ev = halves[k], halves[1 - k]
        t, _ = tune_threshold(score[tu], y[tu])
        pred[ev] = (score[ev] >= t).astype(int)
    return pred


def crossfit_cascade(s1, s3, y, halves, budget):
    pred = np.zeros(len(y), dtype=int)
    esc = np.zeros(len(y), dtype=bool)
    for k in range(2):
        tu, ev = halves[k], halves[1 - k]
        t3, _ = tune_threshold(s3[tu], y[tu])
        lo, hi, _, _ = choose_band(s1[tu], (s3[tu] >= t3).astype(int), y[tu], budget)
        p, e = cascade_predict(s1[ev], (s3[ev] >= t3).astype(int), lo, hi)
        pred[ev], esc[ev] = p, e
    return pred, esc


def offline(args):
    lat_path = RESULTS / "latency_single.csv"
    lat = pd.read_csv(lat_path).set_index("system").mean_ms if lat_path.exists() else None
    rows, preds_all = [], []
    for mode in ("random", "grouped"):
        for seed in [42, 123, 7, 2024, 99]:
            p = RESULTS / "preds" / f"main_{mode}_s{seed}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            if "B3_score" not in d:
                continue
            y = d.label.values
            perm = np.random.default_rng(seed).permutation(len(d))
            halves = [perm[:len(d) // 2], perm[len(d) // 2:]]
            sys_preds = {}
            for s in ("B2", "v3", "B3"):
                sys_preds[s] = (crossfit_single(d[f"{s}_score"].values, y, halves), None)
            ens = (d.B2_score + d.v3_score + d.B3_score).values / 3
            sys_preds["avg(B2,v3,B3)"] = (crossfit_single(ens, y, halves), None)
            for first in ("B2", "v3"):
                for b in BUDGETS:
                    pr, esc = crossfit_cascade(d[f"{first}_score"].values, d.B3_score.values, y, halves, b)
                    sys_preds[f"{first}>B3@{int(b * 100)}"] = (pr, esc)
            for name, (pr, esc) in sys_preds.items():
                m = metrics(y, pr)
                rate = float(esc.mean()) if esc is not None else np.nan
                if lat is not None:
                    if ">" in name:
                        first = name.split(">")[0]
                        exp_lat = lat[first] + rate * lat["B3"]
                    elif name.startswith("avg"):
                        exp_lat = lat["B2"] + lat["v3"] + lat["B3"]
                    else:
                        exp_lat = lat[name]
                else:
                    exp_lat = np.nan
                m.update({"mode": mode, "seed": seed, "system": name, "escalation_rate": rate,
                          "expected_mean_latency_ms": exp_lat})
                rows.append(m)
                for ref in ("B2", "B3"):
                    if name != ref:
                        b_, c_, pm = mcnemar_exact(y, pr, sys_preds[ref][0])
                        preds_all.append({"mode": mode, "seed": seed, "a": name, "b": ref,
                                          "a_right_b_wrong": b_, "a_wrong_b_right": c_, "mcnemar_p": pm,
                                          "f1_a": m["f1"], "f1_b": f1_score(y, sys_preds[ref][0])})
    per = pd.DataFrame(rows)
    per.to_csv(RESULTS / "cascade_offline_per_seed.csv", index=False)
    tests = pd.DataFrame(preds_all)
    agg = []
    for (mode, a, b), g in tests.groupby(["mode", "a", "b"], sort=False):
        md, pt = paired_t(g.f1_a.values, g.f1_b.values)
        agg.append({"mode": mode, "a": a, "b": b, "n_seeds": len(g), "f1_diff_mean": md, "paired_t_p": pt,
                    "mcnemar_p_per_seed": " ".join(f"{x:.3f}" for x in g.mcnemar_p)})
    pd.concat([tests, pd.DataFrame(agg)]).to_csv(RESULTS / "cascade_offline_tests.csv", index=False)
    summ = per.groupby(["mode", "system"], sort=False).agg(
        n_seeds=("seed", "nunique"), f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        precision=("precision", "mean"), recall=("recall", "mean"), fpr=("fpr", "mean"),
        escalation=("escalation_rate", "mean"), exp_latency_ms=("expected_mean_latency_ms", "mean")).reset_index()
    summ["f1_ci95"] = [t_ci95(per[(per["mode"] == r["mode"]) & (per.system == r.system)].f1.values)
                       for _, r in summ.iterrows()]
    summ.to_csv(RESULTS / "cascade_offline_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("##### offline (2-fold cross-fitted on test predictions) #####")
    print(summ.round(4).to_string(index=False))
    print("\n##### paired tests (across seeds) #####")
    print(pd.DataFrame(agg).round(4).to_string(index=False))


# ======================================================================
# live: seed-42 models, validation-tuned band, full robustness protocol
# ======================================================================
class SanitizedInput(Detector):
    """B3 fed the sanitized text (decoded payloads appended)."""

    def __init__(self, base):
        from src.sanitizer import sanitize
        self._san = sanitize
        self.base = base
        self.name = base.name + "+san"
        self.threshold = base.threshold

    def fit(self, *a):
        return self

    def scores(self, texts):
        return self.base.scores([self._san(t) for t in texts])

    def score_one(self, text):
        return self.base.score_one(self._san(text))


def live(args):
    import torch
    from src.data import load_pool, make_split
    from src.trained import load_many
    from src.attacks import attack_sample, benign_sentences
    from run_obfuscation import TECHNIQUES, b64, benign_hash_url
    from run_adaptive import attack_one, BUDGETS as ADV_BUDGETS

    base = load_many(["B2", "v3", "B3"])
    if len(base) < 3:
        raise SystemExit("need results/models/B2_seed42.pkl, v3_seed42.pkl and b3_seed42")
    pool = load_pool(42)
    _, val, test = make_split(pool, 42, "random")
    va_x, va_y = val.text.tolist(), val.label.values
    te_x, te_y = test.text.tolist(), test.label.values

    systems = {"B2": base["B2"], "v3": base["v3"], "B3": base["B3"]}
    b3s = SanitizedInput(base["B3"])
    systems["B3+san"] = b3s
    configs = [("B2", base["B3"], 0.1, False), ("B2", base["B3"], 0.2, False), ("B2", base["B3"], 0.3, False),
               ("B2", base["B3"], 0.2, True), ("v3", base["B3"], 0.2, False), ("v3", base["B3"], 0.2, True)]
    band_rows = []
    for first, second, b, san in configs:
        c = CascadeDetector(base[first], second, max_escalation=b, sanitize_second=san)
        c.fit(None, None, va_x, va_y)
        systems[c.name] = c
        band_rows.append({"system": c.name, "lo": c.lo, "hi": c.hi, "val_f1": c.val_f1,
                          "val_escalation": c.val_escalation})
        print(f"  {c.name:16s} band=[{c.lo:.2f},{c.hi:.2f})  val F1={c.val_f1:.4f}  val escalation={c.val_escalation:.3f}",
              flush=True)
    pd.DataFrame(band_rows).to_csv(RESULTS / "cascade_live_bands.csv", index=False)

    # ---------------- accuracy + latency ----------------
    rng = np.random.default_rng(42)
    lat_texts = [te_x[i] for i in rng.choice(len(te_x), min(512, len(te_x)), replace=False)]
    acc_rows = []
    for name, det in systems.items():
        pr = det.predict(te_x)
        m = metrics(te_y, pr)
        m["system"] = name
        m["test_escalation"] = getattr(det, "last_escalation_rate", np.nan)
        for t in lat_texts[:20]:
            det.score_one(t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        L = []
        for t in lat_texts:
            t0 = time.perf_counter()
            det.score_one(t)
            L.append((time.perf_counter() - t0) * 1000)
        L = np.array(L)
        m.update({"lat_mean_ms": L.mean(), "lat_median_ms": np.median(L), "lat_p95_ms": np.percentile(L, 95)})
        acc_rows.append(m)
        print(f"  {name:16s} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} FPR={m['fpr']:.4f} "
              f"esc={m['test_escalation']:.3f} | latency mean {m['lat_mean_ms']:.1f} median {m['lat_median_ms']:.1f} "
              f"p95 {m['lat_p95_ms']:.1f} ms", flush=True)
    pd.DataFrame(acc_rows).to_csv(RESULTS / "cascade_live_accuracy_latency.csv", index=False)

    # ---------------- obfuscation ----------------
    pos = [t for t, y in zip(te_x, te_y) if y == 1]
    neg = [t for t, y in zip(te_x, te_y) if y == 0]
    obf_rows, ctrl_rows = [], []
    for name, det in systems.items():
        basep = [t for t, p in zip(pos, det.predict(pos)) if p]
        for tech, fn in TECHNIQUES.items():
            k = int(det.predict([fn(t) for t in basep]).sum()) if basep else 0
            obf_rows.append({"system": name, "technique": tech, "n_clean_detected": len(basep), "retained": k,
                             "retention": k / max(len(basep), 1)})
        for cname, texts in [("benign_asis", neg), ("benign_base64_wrapped", [b64(t) for t in neg]),
                             ("benign_with_url_and_hash", [benign_hash_url(t) for t in neg])]:
            fp = int(det.predict(texts).sum())
            ctrl_rows.append({"system": name, "control": cname, "fpr": fp / len(texts)})
        print(f"  obfuscation done: {name}", flush=True)
    obf = pd.DataFrame(obf_rows)
    obf.to_csv(RESULTS / "cascade_live_obfuscation.csv", index=False)
    pd.DataFrame(ctrl_rows).to_csv(RESULTS / "cascade_live_benign_controls.csv", index=False)

    # ---------------- paraphrase ----------------
    para_rows = []
    gp = RESULTS / "paraphrase_generations.csv"
    if gp.exists():
        pg = pd.read_csv(gp).fillna("")
        pg = pg[pg.auto_invalid == ""]
        origs = pg.drop_duplicates("original").original.tolist()
        for name, det in systems.items():
            po = dict(zip(origs, det.predict(origs)))
            g2 = pg.assign(o=pg.original.map(po), p=det.predict(pg.paraphrase.tolist()))
            ev = [1 - h.p[h.o == 1].mean() for _, h in g2.groupby("sample_seed") if (h.o == 1).any()]
            para_rows.append({"system": name, "evasion_given_detected_mean": float(np.mean(ev)),
                              "evasion_given_detected_std": float(np.std(ev, ddof=1)) if len(ev) > 1 else 0.0})
        pd.DataFrame(para_rows).to_csv(RESULTS / "cascade_live_paraphrase.csv", index=False)

    # ---------------- multilingual ----------------
    ml_rows = []
    mp = RESULTS / "multilingual_translations.csv"
    if mp.exists():
        md = pd.read_csv(mp).fillna("")
        for name, det in systems.items():
            for lang, g in md.groupby("lang", sort=False):
                m = metrics(g.label.values, det.predict(g.text.tolist()))
                ml_rows.append({"system": name, "lang": lang, "recall": m["recall"], "fpr": m["fpr"], "f1": m["f1"]})
        pd.DataFrame(ml_rows).to_csv(RESULTS / "cascade_live_multilingual.csv", index=False)

    # ---------------- adaptive attack ----------------
    prompts = attack_sample(150)
    benign_pool = benign_sentences()
    para = {}
    if gp.exists():
        for _, r in pg.iterrows():
            para.setdefault(r.original, []).append(r.paraphrase)
    ops = ["synonym", "benign_padding", "char"] + (["paraphrase"] if para else [])
    adv_rows = []
    for name, det in systems.items():
        rng_a = np.random.default_rng(0)
        flagged = det.predict(prompts)
        succ = []
        for text, f in zip(prompts, flagged):
            if f:
                q, _, _ = attack_one(det, text, rng_a, 20, ops, benign_pool, para.get(text, []))
                succ.append(q)
        for b in ADV_BUDGETS:
            k = sum(1 for q in succ if q is not None and q <= b)
            adv_rows.append({"system": name, "budget": b, "n_flagged": len(succ), "asr": k / max(len(succ), 1)})
        print(f"  adaptive {name:16s} ASR@20={adv_rows[-1]['asr']:.3f} (n={len(succ)})", flush=True)
    adv = pd.DataFrame(adv_rows)
    adv.to_csv(RESULTS / "cascade_live_adaptive.csv", index=False)

    # ---------------- summary ----------------
    pd.set_option("display.width", 250)
    print("\n##### accuracy + latency (seed 42, test) #####")
    print(pd.DataFrame(acc_rows)[["system", "f1", "precision", "recall", "fpr", "test_escalation",
                                   "lat_mean_ms", "lat_median_ms", "lat_p95_ms"]].round(4).to_string(index=False))
    print("\n##### obfuscation retention (%) #####")
    print((obf.pivot(index="technique", columns="system", values="retention") * 100).round(1).to_string())
    print("\n##### benign controls FPR (%) #####")
    print((pd.DataFrame(ctrl_rows).pivot(index="control", columns="system", values="fpr") * 100).round(1).to_string())
    if para_rows:
        print("\n##### paraphrase evasion (auto-valid) #####")
        print(pd.DataFrame(para_rows).round(4).to_string(index=False))
    if ml_rows:
        print("\n##### multilingual F1 / FPR #####")
        mlr = pd.DataFrame(ml_rows)
        print(mlr.pivot(index="system", columns="lang", values="f1").round(3).to_string())
        print(mlr.pivot(index="system", columns="lang", values="fpr").round(3).to_string())
    print("\n##### adaptive ASR by query budget #####")
    print(adv.pivot(index="system", columns="budget", values="asr").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["offline", "live"])
    args = ap.parse_args()
    offline(args) if args.part == "offline" else live(args)


if __name__ == "__main__":
    main()
