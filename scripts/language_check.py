"""
Language-shortcut check: share of attacks among German-looking training
prompts (>= 3 common German function words) versus the overall attack share,
for the seed-42 random split. Writes results/language_check.json.

    py scripts/language_check.py
"""
import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_pool, make_split

DE = re.compile(r"\b(und|nicht|ich|ist|die|der|das|du|bitte|sind|wie|was|mit)\b", re.I)


def main():
    tr, _, _ = make_split(load_pool(42), 42, "random")
    de = tr.text.map(lambda t: len(DE.findall(t)) >= 3)
    out = {"n_train": int(len(tr)), "n_german_like": int(de.sum()),
           "attack_share_german_like": float(tr[de].label.mean()),
           "attack_share_overall": float(tr.label.mean()),
           "german_like_by_source_label": {f"{s}|{l}": int(n) for (s, l), n in
                                           tr[de].groupby(["source", "label"]).size().items()}}
    (ROOT / "results" / "language_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
