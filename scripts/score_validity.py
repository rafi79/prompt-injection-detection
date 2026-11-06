"""
Summarise a filled-in manual validity sheet (intent_preserved = 1/0).

    py scripts/score_validity.py results/adaptive_validity_sheet.csv
    py scripts/score_validity.py results/paraphrase_validity_sheet.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evalstats import wilson


def read_sheet(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue


def main():
    path = Path(sys.argv[1])
    s = read_sheet(path)
    s = s[s.intent_preserved.astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])].copy()
    if s.empty:
        raise SystemExit("No row has intent_preserved = 1 or 0 yet. Fill the column, save as CSV UTF-8, close Excel, rerun.")
    s["intent_preserved"] = s.intent_preserved.astype(float).astype(int)
    rows = []
    groups = [("all", s)] + ([(f"system={k}", g) for k, g in s.groupby("system")] if "system" in s else []) \
        + ([(f"operator={k}", g) for k, g in s.groupby("final_operator")] if "final_operator" in s else [])
    for name, g in groups:
        k, n = int(g.intent_preserved.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"subset": name, "valid": k, "n": n, "rate": k / n, "wilson_low": lo, "wilson_high": hi})
    out = pd.DataFrame(rows)
    dest = path.with_name(path.stem + "_summary.csv")
    out.to_csv(dest, index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
