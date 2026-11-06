"""
Generate every LaTeX table, plot coordinate list and in-text number used by
the paper directly from the CSV files in results/. Nothing in the paper is
typed by hand: re-running this script after new experiments updates it.

    py scripts/make_tables.py            # writes paper/generated/*.tex
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "generated"

NAMES = {"B1": "B1 keyword", "v1": "v1 lexical", "v2": "v2 semantic-lite", "v2cf": "v2cf cross-fitted",
         "v3": "v3 semantic-lite+", "B2": "B2 TF-IDF+SVM", "B3": "B3 PromptGuard (ft)",
         "B3raw": "B3raw PromptGuard", "LenOnly": "Length only"}
MAIN_ORDER = ["B1", "v1", "v2", "v2cf", "v3", "B2", "B3", "B3raw", "LenOnly"]
macros = {}


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


_DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
        "7": "Seven", "8": "Eight", "9": "Nine"}


def mac(name, value):
    """LaTeX macro names may only contain letters: digits become words,
    everything else is dropped (e.g. 'casB2>B3@20F' -> 'casBTwotoBThreeatTwoZeroF')."""
    clean = "".join(_DIG.get(ch, ch) for ch in name if ch.isalnum())
    if clean in macros and macros[clean] != value:
        print(f"[warn] macro collision {clean}: {macros[clean]} vs {value}")
    macros[clean] = value


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", OUT / name)


def read(name, **kw):
    p = RES / name
    return pd.read_csv(p, **kw) if p.exists() else None


# ------------------------------------------------------------------ data --
def dataset():
    r = json.loads((RES / "dataset_report.json").read_text())
    bs = r["by_source"]
    tr = r["split_seed42_random"]
    import re
    nums = [int(x) for x in re.findall(r"(\d+) pos / (\d+) neg", tr) for x in x]
    mac("nTotal", f"{r['n_total']:,}".replace(",", "{,}"))
    mac("nPos", f"{r['n_pos']:,}".replace(",", "{,}"))
    mac("nNeg", f"{r['n_neg']:,}".replace(",", "{,}"))
    mac("nConflicts", str(r["label_conflicts_resolved_to_positive"]))
    mac("nPrDup", str(r["pr1m8_rows_duplicating_other_source"]))
    mac("nPrRaw", str(r["pr1m8_rows_raw"]))
    mac("nPrUnique", str(r["pr1m8_rows_raw"] - r["pr1m8_rows_duplicating_other_source"]))
    mac("nNearDupTexts", f"{r['texts_in_multi_member_groups']:,}".replace(",", "{,}"))
    mac("nNearDupMixed", str(r["groups_with_both_labels"]))
    mac("nLargestGroup", str(r["largest_group"]))
    mac("fingerprint", r["fingerprint"])
    t = r"""\begin{table}[t]
\centering
\caption{Cleaned dataset (seed-42 random split; benign capped at $3\times$ attacks).}
\label{tab:dataset}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Split} & \textbf{Injection} & \textbf{Benign} \\
\midrule
Training & %d & %d \\
Validation & %d & %d \\
Test & %d & %d \\
\midrule
\multicolumn{3}{l}{\footnotesize JailbreakHub %d/%d; deepset %d/%d; pr1m8 %d/0 (attack/benign).}\\
\bottomrule
\end{tabular}
\end{table}
""" % (nums[0], nums[1], nums[2], nums[3], nums[4], nums[5],
       bs.get("jailbreak_llms|label=1", 0), bs.get("jailbreak_llms|label=0", 0),
       bs.get("deepset|label=1", 0), bs.get("deepset|label=0", 0), bs.get("pr1m8|label=1", 0))
    write("tab_dataset.tex", t)


# ------------------------------------------------------------------ main --
def main_table():
    r = read("main_random_summary.csv").set_index("system")
    g = read("main_grouped_summary.csv").set_index("system")
    lm = read("main_random_length_matched_summary.csv", header=[0, 1], index_col=0)
    rows = []
    for s in MAIN_ORDER:
        if s not in r.index:
            continue
        x = r.loc[s]
        gf = (f"{g.loc[s].f1_mean:.3f}$\\pm${g.loc[s].f1_ci95:.3f}" + ("$^\\dagger$" if g.loc[s].n_seeds < 5 else "")) \
            if s in g.index else "--"
        lauc = f3(lm.loc[s, ("roc_auc", "mean")]) if s in lm.index else "--"
        rows.append(f"{NAMES[s]} & {x.f1_mean:.3f}$\\pm${x.f1_ci95:.3f} & {x.precision_mean:.3f} & "
                    f"{x.recall_mean:.3f} & {f3(x.roc_auc_mean)} & {lauc} & {gf} \\\\")
        for split, df in (("R", r), ("G", g)):
            if s in df.index:
                mac(f"f{split}{s.replace('3raw', 'raw')}", f"{df.loc[s].f1_mean:.3f}")
                mac(f"ci{split}{s.replace('3raw', 'raw')}", f"{df.loc[s].f1_ci95:.3f}")
        mac(f"auc{s.replace('3raw', 'raw')}", f3(x.roc_auc_mean))
    t = r"""\begin{table*}[t]
\centering
\caption{Test performance, mean over seeds ($\pm$ = 95\% $t$-interval of F1). Random split: 5 seeds. Grouped split (no near-duplicate group crosses splits): 5 seeds, $^\dagger$3 seeds. LM-AUC: ROC-AUC on a length-matched, class-balanced test subset. Flagging every prompt gives F1 = 0.400.}
\label{tab:main}
\small
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{5}{c}{\textbf{Random split}} & \textbf{Grouped split} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-7}
\textbf{System} & \textbf{F1} & \textbf{Prec.} & \textbf{Rec.} & \textbf{AUC} & \textbf{LM-AUC} & \textbf{F1} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_main.tex", t)

    # paired tests
    for mode in ("random", "grouped"):
        pt = read(f"main_{mode}_paired_tests.csv")
        agg = pt[pt.seed.astype(str) == "all(paired t)"]
        for _, x in agg.iterrows():
            key = f"p{mode[0].upper()}{x.a}{x.b}".replace("3raw", "raw")
            mac(key, f"{x.paired_t_p:.3f}")
            mac("d" + key[1:], f"{x.f1_diff:+.3f}")


# -------------------------------------------------------------- ablation --
def ablation():
    per = read("main_random_per_seed.csv")
    order = [("v2cf", "word", "1", "--"), ("sem:char:1:0:cf", "char", "1", "--"),
             ("sem:both:1:0:cf", "word+char", "1", "--"), ("sem:both:8:0:cf", "word+char", "8", "--"),
             ("sem:word:8:10:cf", "word", "8", "10"), ("v3", "word+char", "8", "10")]
    from scipy import stats
    v3 = per[per.system == "v3"].set_index("seed").f1
    rows = []
    for s, an, k, knn in order:
        f = per[per.system == s].set_index("seed").f1
        ci = stats.sem(f) * stats.t.ppf(0.975, len(f) - 1)
        if s != "v3":
            common = sorted(set(f.index) & set(v3.index))
            p = stats.ttest_rel(f.loc[common], v3.loc[common]).pvalue
            ptxt = f"{p:.3f}"
        else:
            ptxt = "--"
        auc = per[per.system == s].roc_auc.mean()
        rows.append(f"{an} & {k} & {knn} & {f.mean():.3f}$\\pm${ci:.3f} & {auc:.3f} & {ptxt} \\\\")
    t = r"""\begin{table}[t]
\centering
\caption{Semantic-stage ablation (random split, 5 seeds; all cross-fitted). $p$: paired $t$-test against v3.}
\label{tab:ablation}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccc}
\toprule
\textbf{TF-IDF} & \textbf{Prot.} & \textbf{kNN} & \textbf{F1} & \textbf{AUC} & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_ablation.tex", t)


# --------------------------------------------------------------- latency --
def latency():
    s = read("latency_single.csv").set_index("system")
    b = read("latency_batch.csv").pivot(index="system", columns="batch_size", values="per_item_ms_median")
    st = read("latency_stages.csv")
    hw = json.loads((RES / "hardware.json").read_text())
    mac("gpuName", hw.get("gpu", "GPU").replace("NVIDIA ", ""))
    rows = []
    for sys_ in ["B1", "v1", "v2", "v2cf", "v3", "B2", "B3"]:
        x = s.loc[sys_]
        dev = "GPU" if x.device == "cuda" else "CPU"
        rows.append(f"{sys_} & {dev} & {x.median_ms:.1f} & {x.mean_ms:.1f} & {x.p95_ms:.1f} & "
                    f"{x.throughput_single_stream_req_s:.0f} & {x.model_size_mb:.1f} & "
                    f"{b.loc[sys_, 32]:.1f} \\\\")
        mac(f"latMed{sys_}", f"{x.median_ms:.1f}")
        mac(f"latMean{sys_}", f"{x.mean_ms:.1f}")
        mac(f"latPn{sys_}", f"{x.p95_ms:.1f}")
        mac(f"size{sys_}", f"{x.model_size_mb:.1f}")
    mac("latNtexts", str(int(s.n_texts.iloc[0])))
    t = r"""\begin{table}[t]
\centering
\caption{Single-request latency (ms) over the same %d test prompts (3 timed passes after warm-up), measured single-stream throughput (req/s), model size (MB), and per-item latency at batch size 32.}
\label{tab:latency}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{\textbf{Latency (ms)}} & & & \textbf{bs=32} \\
\cmidrule(lr){3-5}
\textbf{Sys.} & \textbf{Dev.} & \textbf{Med.} & \textbf{Mean} & \textbf{p95} & \textbf{req/s} & \textbf{MB} & \textbf{ms} \\
\midrule
""" % int(s.n_texts.iloc[0]) + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_latency.tex", t)
    if st is not None:
        v3 = st[st.system == "v3"].iloc[0]
        mac("stageSanitize", f"{v3.sanitize:.1f}")
        mac("stageKeyword", f"{v3.keyword:.1f}")
        mac("stageSemanticVthree", f"{v3.semantic:.1f}")
        v2 = st[st.system == "v2"].iloc[0]
        mac("stageSemanticVtwo", f"{v2.semantic:.1f}")


# --------------------------------------------------------------- cascade --
def cascade():
    c = read("cascade_offline_summary.csv")
    if c is None:
        return
    tests = read("cascade_offline_tests.csv")
    agg = tests[tests.n_seeds.notna()] if "n_seeds" in tests else tests
    show = ["B2", "v3", "B3", "B2>B3@10", "B2>B3@20", "B2>B3@30", "v3>B3@20", "avg(B2,v3,B3)"]
    label = {"avg(B2,v3,B3)": "Avg(B2,v3,B3)"}
    rows = []
    for s in show:
        r = c[(c["mode"] == "random") & (c.system == s)]
        g = c[(c["mode"] == "grouped") & (c.system == s)]
        if r.empty:
            continue
        r = r.iloc[0]

        def p_vs(ref, mode="random"):
            x = agg[(agg["mode"] == mode) & (agg.a == s) & (agg.b == ref)]
            return f"{x.paired_t_p.iloc[0]:.3f}" if len(x) else "--"
        esc = "--" if np.isnan(r.escalation) else pct(r.escalation)
        gtxt = f"{g.iloc[0].f1_mean:.3f}" if len(g) else "--"
        rows.append(f"{label.get(s, s.replace('>', '$\\rightarrow$'))} & {r.f1_mean:.3f}$\\pm${r.f1_ci95:.3f} & "
                    f"{p_vs('B2')} & {p_vs('B3')} & {gtxt} & {esc} & {r.exp_latency_ms:.1f} \\\\")
        key = s.replace(">", "to").replace("@", "at")
        mac(f"cas{key}F", f"{r.f1_mean:.3f}")
        mac(f"cas{key}Esc", esc)
        mac(f"cas{key}Lat", f"{r.exp_latency_ms:.1f}")
        mac(f"cas{key}pBtwo", p_vs("B2"))
        mac(f"cas{key}pBthree", p_vs("B3"))
    t = r"""\begin{table}[t]
\centering
\caption{Cascades (stage 1 on CPU, PromptGuard only inside the uncertainty band; @$x$: at most $x$\% escalated). Thresholds and bands chosen by 2-fold cross-fitting on test predictions; random split 5 seeds, grouped 3 seeds. $p$: paired $t$-test of F1 against B2 / B3. Esc.: escalated share (\%). Lat.: expected mean latency from Table~\ref{tab:latency}.}
\label{tab:cascade}
\small
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccccc}
\toprule
 & \textbf{F1 (R)} & $p_{B2}$ & $p_{B3}$ & \textbf{F1 (G)} & \textbf{Esc.} & \textbf{Lat.} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_cascade.tex", t)

    # live cascade (seed 42) if available
    live = read("cascade_live_accuracy_latency.csv")
    if live is not None:
        adv = read("cascade_live_adaptive.csv")
        obf = read("cascade_live_obfuscation.csv")
        para = read("cascade_live_paraphrase.csv")
        ml = read("cascade_live_multilingual.csv")
        rows = []
        for _, x in live.iterrows():
            asr = adv[(adv.system == x.system) & (adv.budget == 20)].asr.iloc[0] if adv is not None else np.nan
            enc = obf[(obf.system == x.system) & (obf.technique.isin(["base64", "hex", "base64_double"]))].retention.mean() \
                if obf is not None else np.nan
            pe = para[para.system == x.system].evasion_given_detected_mean.iloc[0] if para is not None else np.nan
            mlf = ml[(ml.system == x.system) & (ml.lang != "en")].f1.mean() if ml is not None else np.nan
            esc = "--" if np.isnan(x.test_escalation) else pct(x.test_escalation)
            rows.append(f"{x.system.replace('>', '$\\rightarrow$')} & {x.f1:.3f} & {esc} & {x.lat_mean_ms:.1f} & "
                        f"{x.lat_p95_ms:.1f} & {pct(enc)} & {pct(pe)} & {pct(asr)} & {f3(mlf)} \\\\")
        t = r"""\begin{table*}[t]
\centering
\caption{Seed-42 deployment check with the band chosen on the validation split: test F1, escalated share (\%), measured single-request latency (ms), mean retention of encoded attacks (Base64, double Base64, hex; \%), paraphrase evasion (\%), adaptive-attack success at 20 queries (\%), and mean F1 over the six non-English languages. +san: PromptGuard receives the sanitized text.}
\label{tab:cascadelive}
\small
\begin{tabular}{lcccccccc}
\toprule
\textbf{System} & \textbf{F1} & \textbf{Esc.} & \textbf{Mean ms} & \textbf{p95 ms} & \textbf{Encoded} & \textbf{Para. ev.} & \textbf{ASR@20} & \textbf{Non-EN F1} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
        write("tab_cascade_live.tex", t)


# ------------------------------------------------------------------ LOSO --
def loso():
    s = read("loso_summary.csv")
    per = read("loso_per_seed.csv")
    rows = []
    for sys_ in ["B1", "v1", "v2", "v2cf", "v3", "B2", "B3"]:
        def get(h, m):
            x = s[(s.heldout == h) & (s.system == sys_)]
            return x[m].iloc[0] if len(x) else np.nan
        seed42 = per[(per.system == sys_) & (per.seed == 42)]

        def g42(h, m):
            x = seed42[seed42.heldout == h]
            return x[m].iloc[0] if len(x) else np.nan
        nseeds = int(s[(s.system == sys_)].n_seeds.max())
        tag = "$^\\ddagger$" if nseeds == 1 else ""
        dup = s[(s.heldout.str.startswith("pr1m8_seen")) & (s.system == sys_)].recall_mean
        rows.append(f"{sys_}{tag} & {f3(get('jailbreak_llms', 'f1_mean'))} / {pct(get('jailbreak_llms', 'fpr_mean'), 0)} / "
                    f"{f3(get('jailbreak_llms', 'roc_auc_mean'))} & {f3(get('deepset', 'f1_mean'))} / "
                    f"{pct(get('deepset', 'fpr_mean'), 0)} & {pct(get('pr1m8', 'recall_mean'), 0)} & "
                    f"{pct(dup.iloc[0], 0) if len(dup) else '--'} \\\\")
        for h, k in (("jailbreak_llms", "JB"), ("deepset", "DS")):
            mac(f"loso{k}F{sys_}", f3(get(h, "f1_mean")))
            mac(f"loso{k}Auc{sys_}", f3(get(h, "roc_auc_mean")))
            mac(f"loso{k}Fpr{sys_}", pct(get(h, "fpr_mean"), 0))
        mac(f"losoPR{sys_}", pct(get("pr1m8", "recall_mean"), 0))
        if len(dup):
            mac(f"losoDup{sys_}", pct(dup.iloc[0], 0))
    t = r"""\begin{table}[t]
\centering
\caption{Leave-one-source-out (train and tune on the other two sources). JailbreakHub: F1 / FPR\% / AUC; deepset: F1 / FPR\%; pr1m8: recall (\%) on its 31 unique prompts and on the 51 prompts duplicated from deepset (seen in training). Mean of 3 seeds; $^\ddagger$seed 42 only.}
\label{tab:loso}
\small
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcccc}
\toprule
 & \textbf{JailbreakHub} & \textbf{deepset} & \multicolumn{2}{c}{\textbf{pr1m8 recall}} \\
\cmidrule(lr){4-5}
\textbf{Sys.} & F1 / FPR / AUC & F1 / FPR & unique & dup. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_loso.tex", t)
    cat = read("loso_pr1m8_categories_summary.csv")
    if cat is not None:
        m = cat[(cat.subset == "pr1m8_unique") & (cat.category == "Multilingual")].set_index("system")
        for sys_ in m.index:
            mac(f"prMulti{sys_}", f"{m.loc[sys_].detected / m.loc[sys_].seeds:.1f}")


# ----------------------------------------------------------- obfuscation --
def obfuscation():
    o = read("obfuscation_suite.csv")
    c = read("obfuscation_benign_controls.csv")
    techs = [("zero_width", "Zero-width"), ("homoglyph", "Homoglyph"), ("fullwidth", "Full-width"),
             ("base64", "Base64 (wrapped)"), ("base64_urlsafe_nopad", "Base64 URL-safe"),
             ("base64_double", "Base64 double"), ("hex", "Hex"), ("rot13", "ROT13"), ("reversed", "Reversed"),
             ("leetspeak", "Leetspeak"), ("spaced_letters", "Spaced letters")]
    cols = [("B1", "legacy"), ("B1", "full"), ("v3", "none"), ("v3", "full"), ("B2", "none"), ("B2", "full"),
            ("B3", "model_only")]

    def v(sys_, san, tech):
        x = o[(o.system == sys_) & (o.sanitizer == san) & (o.technique == tech)]
        return pct(x.retention.iloc[0], 0) if len(x) else "--"
    rows = [f"{lab} & " + " & ".join(v(s, sn, t) for s, sn in cols) + r" \\" for t, lab in techs]

    def fp(sys_, san, ctrl):
        x = c[(c.system == sys_) & (c.sanitizer == san) & (c.control == ctrl)]
        return pct(x.fpr.iloc[0], 0) if len(x) else "--"
    rows.append(r"\midrule")
    for ctrl, lab in (("benign_asis", "Benign FPR"), ("benign_base64_wrapped", "Benign, Base64-wrapped"),
                      ("benign_with_url_and_hash", "Benign + URL/hash")):
        rows.append(f"{lab} & " + " & ".join(fp(s, sn, ctrl) for s, sn in cols) + r" \\")
    for t, _ in techs:
        mac("obfLegacy" + "".join(ch for ch in t if ch.isalpha()), v("B1", "legacy", t))
    mac("obfBthreeBase", v("B3", "model_only", "base64"))
    mac("obfBthreeHex", v("B3", "model_only", "hex"))
    t = r"""\begin{table}[t]
\centering
\caption{Retention (\%) of attacks each system detects in clean form after one obfuscation, with the original (orig.), no (none) or the revised (rev.) sanitizer; bottom: false-positive rate (\%) on benign test prompts and on two constructed benign controls. PromptGuard (B3) has no sanitizer.}
\label{tab:obf}
\small
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{B1}} & \multicolumn{2}{c}{\textbf{v3}} & \multicolumn{2}{c}{\textbf{B2}} & \textbf{B3} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\textbf{Technique} & orig. & rev. & none & rev. & none & rev. & -- \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_obf.tex", t)


# ------------------------------------------------------- attack summary --
def attacks():
    para = read("paraphrase_results.csv")
    adv = read("adaptive_asr.csv")
    uni = read("universal_suffix.csv").set_index("system")
    cost = read("adaptive_window_cost.csv").set_index("system")
    rows = []
    for sys_ in ["B1", "v1", "v2", "v2cf", "v3", "B2", "B3", "v2cf+win", "v3+win", "B2+win"]:
        p = para[(para.system == sys_) & (para.subset == "auto_valid")]
        pe = f"{pct(p.evasion_given_detected_mean.iloc[0])}" if len(p) else "--"
        a = adv[adv.system == sys_].set_index("budget").asr
        u = uni.loc[sys_] if sys_ in uni.index else None
        ut = "--" if (u is None or sys_ == "B1") else f"{pct(u.evasion_learned_suffix, 0)} / {pct(u.evasion_random_suffix_mean, 0)}"
        cf = f3(cost.loc[sys_].f1) if sys_ in cost.index else "--"
        rows.append(f"{sys_} & {pe} & {pct(a.get(5, np.nan), 0)} & {pct(a.get(20, np.nan), 0)} & {ut} & {cf} \\\\")
        key = sys_.replace("+win", "win")
        mac(f"asr{key}", pct(a.get(20, np.nan), 0))
        mac(f"para{key}", pe)
        if u is not None and sys_ != "B1":
            mac(f"uni{key}", pct(u.evasion_learned_suffix, 0))
            mac(f"uniRand{key}", pct(u.evasion_random_suffix_mean, 0))
        if sys_ in cost.index:
            mac(f"winF{key}", f3(cost.loc[sys_].f1))
            mac(f"winFpr{key}", pct(cost.loc[sys_].fpr))
    t = r"""\begin{table}[t]
\centering
\caption{Adaptive robustness (seed-42 models, 150 test attacks). Para.: evasion (\%) among detected attacks after one LLM paraphrase (auto-valid, mean of 3 samples). ASR: black-box attack success (\%) at 5 and 20 queries. Suffix: held-out evasion (\%) with the learned / a random 8-word suffix. Clean F1 of the windowed (+win) variants on the test split.}
\label{tab:attacks}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Sys.} & \textbf{Para.} & \textbf{ASR@5} & \textbf{ASR@20} & \textbf{Suffix} & \textbf{F1} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_attacks.tex", t)
    ops = read("adaptive_examples.csv").final_operator.value_counts()
    mac("advPadding", str(int(ops.get("benign_padding", 0))))
    mac("advSuccessTotal", str(int(ops.sum())))
    pv = read("paraphrase_validity_summary.csv")
    if pv is not None:
        mac("paraValid", f"{int(pv.valid.iloc[0])}/{int(pv.n.iloc[0])}")
        mac("paraValidLo", pct(pv.wilson_low.iloc[0], 0))
        mac("paraValidHi", pct(pv.wilson_high.iloc[0], 0))
    av = read("adaptive_validity_sheet_summary.csv")
    if av is not None:
        a0 = av[av.subset == "all"].iloc[0]
        mac("advValid", f"{int(a0.valid)}/{int(a0.n)}")
        mac("advValidLo", pct(a0.wilson_low, 0))
        mac("advValidHi", pct(a0.wilson_high, 0))
        pp = av[av.subset == "operator=paraphrase"]
        if len(pp):
            mac("advValidPara", f"{int(pp.valid.iloc[0])}/{int(pp.n.iloc[0])}")
        bp = av[av.subset == "operator=benign_padding"]
        if len(bp):
            mac("advValidPad", f"{int(bp.valid.iloc[0])}/{int(bp.n.iloc[0])}")
    g = read("paraphrase_generations.csv")
    if g is not None:
        flags = g.auto_invalid.fillna("").value_counts()
        mac("paraAutoInvalid", str(int(len(g) - flags.get("", 0))))
        mac("paraTotal", str(len(g)))


# ---------------------------------------------------------- multilingual --
def multilingual():
    m = read("multilingual_results.csv")
    langs = ["en", "es", "fr", "de", "zh", "ar", "bn"]
    rows = []
    for sys_ in ["B1", "v2", "v2cf", "v3", "B2", "B3"]:
        cells = []
        for L in langs:
            x = m[(m.system == sys_) & (m.lang == L)]
            cells.append(f"{x.f1.iloc[0]:.2f}/{pct(x.fpr.iloc[0], 0)}" if len(x) else "--")
        rows.append(f"{sys_} & " + " & ".join(cells) + r" \\")
    for L in langs:
        for sys_ in ["B2", "B3", "v3"]:
            x = m[(m.system == sys_) & (m.lang == L)]
            if len(x):
                mac(f"ml{sys_}{L}F", f"{x.f1.iloc[0]:.2f}")
                mac(f"ml{sys_}{L}Fpr", pct(x.fpr.iloc[0], 0))
    t = r"""\begin{table*}[t]
\centering
\caption{Machine-translated attacks and benign prompts (150 each per language, NLLB-200): F1 / FPR (\%).}
\label{tab:multi}
\small
\begin{tabular}{lccccccc}
\toprule
\textbf{System} & \textbf{English} & \textbf{Spanish} & \textbf{French} & \textbf{German} & \textbf{Chinese} & \textbf{Arabic} & \textbf{Bengali} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_multi.tex", t)
    lc = RES / "language_check.json"
    if lc.exists():
        j = json.loads(lc.read_text())
        mac("deAttackShare", pct(j["attack_share_german_like"], 0))
        mac("allAttackShare", pct(j["attack_share_overall"], 0))
        mac("deN", str(j["n_german_like"]))


# ---------------------------------------------------------------- figures --
def figures():
    """Complete pgfplots \\addplot commands (the axis environments live in the paper)."""
    r = read("main_random_summary.csv").set_index("system")
    s = read("latency_single.csv").set_index("system")
    pts = " ".join(f"({s.loc[x].mean_ms:.2f},{r.loc[x].f1_mean:.4f}) +- (0,{r.loc[x].f1_ci95:.4f}) [{x}]"
                   for x in ["B1", "v1", "v2", "v3", "B2", "B3"])
    out = [r"\addplot[only marks, mark=*, nodes near coords, point meta=explicit symbolic,"
           r" error bars/.cd, y dir=both, y explicit] coordinates {" + pts + "};"]
    c = read("cascade_offline_summary.csv")
    if c is not None:
        cp = []
        for x in ["B2>B3@10", "B2>B3@20", "B2>B3@30"]:
            y = c[(c["mode"] == "random") & (c.system == x)]
            if len(y):
                cp.append(f"({y.exp_latency_ms.iloc[0]:.2f},{y.f1_mean.iloc[0]:.4f}) +- (0,{y.f1_ci95.iloc[0]:.4f})"
                          f" [{x.split('@')[1]}]")
        if cp:
            out += [r"\addplot[only marks, mark=square*, red, nodes near coords, point meta=explicit symbolic,"
                    r" error bars/.cd, y dir=both, y explicit] coordinates {" + " ".join(cp) + "};"]
    write("fig_pareto.tex", "\n".join(out) + "\n")
    a = read("adaptive_asr.csv")
    styles = {"B1": "gray, mark=x", "v2cf": "blue!60, mark=o", "v3": "blue, mark=*", "B2": "orange, mark=square*",
              "B3": "red, mark=triangle*", "v3+win": "teal, dashed, mark=diamond*"}
    lines = []
    for sys_, st in styles.items():
        x = a[a.system == sys_].sort_values("budget")
        if len(x):
            lines.append(r"\addplot[" + st + "] coordinates {" +
                         " ".join(f"({int(b)},{100 * v:.1f})" for b, v in zip(x.budget, x.asr)) + "};")
    write("fig_adaptive.tex", "\n".join(lines) + "\n")


def main():
    for fn in (dataset, main_table, ablation, latency, cascade, loso, obfuscation, attacks, multilingual, figures):
        try:
            fn()
        except Exception as e:
            print(f"[warn] {fn.__name__}: {type(e).__name__}: {e}")
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    write("numbers.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
