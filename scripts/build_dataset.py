"""
Build the cleaned dataset from the pinned raw files and write the seed-42
random split to data/processed/ (for inspection only - every experiment
script rebuilds its splits from src/data.py, so it never depends on these
CSVs being up to date).

    py scripts/fetch_data.py
    py scripts/build_dataset.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_pool, make_split, fingerprint, describe, near_duplicate_groups

OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rep = {}
    pool = load_pool(42, report=rep)
    train, val, test = make_split(pool, 42, "random")
    cols = ["text", "label", "source", "pr1m8_category", "language", "all_sources"]
    for name, d in (("train", train), ("val", val), ("test", test)):
        d[cols].to_csv(OUT / f"{name}.csv", index=False)

    groups = near_duplicate_groups(pool.text.tolist())
    import pandas as pd
    g = pd.Series(groups)
    size = g.map(g.value_counts())
    mixed = pd.DataFrame({"g": groups, "y": pool.label.values}).groupby("g").y.nunique().gt(1).sum()
    rep.update({
        "fingerprint": fingerprint(pool),
        "split_seed42_random": describe(train, val, test),
        "near_duplicate_groups(cos>=0.8)": int(g.nunique()),
        "texts_in_multi_member_groups": int((size > 1).sum()),
        "largest_group": int(size.max()),
        "groups_with_both_labels": int(mixed),
    })
    rep["by_source"] = {f"{k[0]}|label={k[1]}": int(v) for k, v in rep["by_source"].items()}
    print(json.dumps(rep, indent=2))
    (RESULTS / "dataset_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {OUT}/train.csv, val.csv, test.csv and results/dataset_report.json")


if __name__ == "__main__":
    main()
