"""
Multilingual robustness (reviewer R6-e).

The same 150 test attack prompts used by the other attack scripts, plus 150
seeded benign test prompts (so that false positives on non-English text are
measured too), are machine-translated sentence by sentence into Spanish,
French, German, Chinese (simplified), Arabic and Bengali with NLLB-200
(facebook/nllb-200-distilled-600M, CC-BY-NC 4.0, research use). Prompts are
truncated to <= 400 characters before translation.

Every system (trained on the English-dominant training split) is then scored
on each language: recall on attacks, FPR on benign text, F1.

Step 1 (GPU):  py scripts/run_multilingual.py translate
Step 2:        py scripts/run_multilingual.py evaluate --with-b3

Writes results/multilingual_translations.csv and results/multilingual_results.csv
"""
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.attacks import attack_sample
from src.evalstats import metrics

RESULTS = ROOT / "results"
TR = RESULTS / "multilingual_translations.csv"
MT_MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {"es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "zh": "zho_Hans", "ar": "arb_Arab",
         "bn": "ben_Beng"}


def sentences(t):
    parts = [p.strip() for p in re.split(r"(?<=[.!?\n])\s+", t) if p.strip()]
    return parts or [t]


def translate(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MT_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MT_MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    attacks = attack_sample(args.n, limit=400)
    benign = attack_sample(args.n, limit=400, label=0)
    items = [(i, 1, t) for i, t in enumerate(attacks)] + [(i, 0, t) for i, t in enumerate(benign)]
    rows = [{"id": i, "label": y, "lang": "en", "text": t} for i, y, t in items]
    for code, nllb in LANGS.items():
        tgt = tok.convert_tokens_to_ids(nllb)
        for k, (i, y, t) in enumerate(items):
            sents = sentences(t)
            out = []
            for j in range(0, len(sents), 16):
                enc = tok(sents[j:j + 16], return_tensors="pt", padding=True, truncation=True,
                          max_length=200).to(dev)
                with torch.no_grad():
                    gen = model.generate(**enc, forced_bos_token_id=tgt, max_new_tokens=256, num_beams=2)
                out += tok.batch_decode(gen, skip_special_tokens=True)
            rows.append({"id": i, "label": y, "lang": code, "text": " ".join(out)})
        print(f"  translated into {code}", flush=True)
    pd.DataFrame(rows).to_csv(TR, index=False)
    print(f"saved {TR}")


def evaluate(args):
    from src.trained import load_many
    df = pd.read_csv(TR).fillna("")
    dets = load_many(args.systems + (["B3"] if args.with_b3 else []))
    rows = []
    for name, det in dets.items():
        for lang, g in df.groupby("lang", sort=False):
            sc = det.scores(g.text.tolist())
            m = metrics(g.label.values, (sc >= det.threshold).astype(int), sc)
            m.update({"system": name, "lang": lang})
            rows.append(m)
        print(f"  {name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "multilingual_results.csv", index=False)
    pd.set_option("display.width", 220)
    order = ["en"] + list(LANGS)
    for metric in ("recall", "fpr", "f1"):
        print(f"\n##### {metric} by language #####")
        print(out.pivot(index="system", columns="lang", values=metric)[order].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["translate", "evaluate"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--systems", nargs="+", default=["B1", "v1", "v2", "v2cf", "v3", "B2"])
    ap.add_argument("--with-b3", action="store_true")
    args = ap.parse_args()
    translate(args) if args.step == "translate" else evaluate(args)


if __name__ == "__main__":
    main()
