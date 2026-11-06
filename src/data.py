"""
Single source of truth for the dataset used by EVERY experiment.

All scripts call `load_pool(seed)` / `make_split(...)` from here, so every
number in the paper comes from one dataset version, built deterministically
from raw files pinned to exact upstream commits (see scripts/fetch_data.py
and data/raw/MANIFEST.json).

Cleaning rules (documented in the paper):
  1. whitespace-normalised, case-folded exact de-duplication ACROSS sources
     (pr1m8 copies 50 of its 82 examples verbatim from deepset; those rows
     are kept once, attributed to deepset, and flagged `dup_of_other_source`)
  2. texts that appear with BOTH labels (39 JailbreakHub prompts are listed as
     both jailbreak and regular) are kept only as positives
  3. texts of <= 5 characters dropped
  4. benign pool capped at 3x positives (seeded sample)
"""
import json
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# upstream commits the raw files are pinned to (also written to MANIFEST.json)
PINNED = {
    "jailbreak_llms": {
        "repo": "verazuo/jailbreak_llms",
        "sha": "4f4031bf8be187f4478c7f94f42b08714722c12e",
        "files": {
            "data/prompts/jailbreak_prompts_2023_12_25.csv":
                "accf97463de96c33c48453dfee9600191ff6fd5a0c43f3f11d5ebf98d1a5de55",
            "data/prompts/regular_prompts_2023_12_25.csv":
                "fb82e3f88fbd5d9c6edf8927b138510f339b85012cab24d41d13e1665a9c5819",
        },
    },
    "deepset": {
        "repo": "sinanw/llm-security-prompt-injection",
        "sha": "c8af0b015a9a4f3380f9e20214828c2dbeadc72c",
        "files": {
            "data/raw/train-00000-of-00001-9564e8b05b4757ab.parquet":
                "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
            "data/raw/test-00000-of-00001-701d16158af87368.parquet":
                "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
        },
    },
    "pr1m8": {
        "repo": "pr1m8/prompt-injections",
        "sha": "a0ee668c48793f40350489e34727a47d24eeb21f",
        "files": {
            "prompt_injections.csv":
                "008fa5ba7a9f3fc2c4f29cf2c3abe81ed68855afac5b03f914c27f0890a1bd96",
        },
    },
}

SRC_JB, SRC_DS, SRC_PR = "jailbreak_llms", "deepset", "pr1m8"
SOURCES = [SRC_JB, SRC_DS, SRC_PR]


def raw_path(source: str, rel: str) -> Path:
    return RAW / source / rel


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw(strict: bool = True):
    problems = []
    for src, spec in PINNED.items():
        for rel, expected in spec["files"].items():
            p = raw_path(src, rel)
            if not p.exists():
                problems.append(f"missing {p}")
            elif _sha256(p) != expected:
                problems.append(f"hash mismatch {p}")
    if problems and strict:
        raise RuntimeError("Raw data not at pinned version - run scripts/fetch_data.py\n  "
                           + "\n  ".join(problems))
    return problems


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def load_raw_rows() -> pd.DataFrame:
    """All raw rows from the three sources, before any cleaning."""
    verify_raw()
    jb_dir = raw_path(SRC_JB, "data/prompts")
    jb = pd.read_csv(jb_dir / "jailbreak_prompts_2023_12_25.csv")
    reg = pd.read_csv(jb_dir / "regular_prompts_2023_12_25.csv")
    rows = [
        pd.DataFrame({"text": jb["prompt"].astype(str), "label": 1, "source": SRC_JB}),
        pd.DataFrame({"text": reg["prompt"].astype(str), "label": 0, "source": SRC_JB}),
    ]
    ds_dir = raw_path(SRC_DS, "data/raw")
    ds = pd.concat([pd.read_parquet(p) for p in sorted(ds_dir.glob("*.parquet"))], ignore_index=True)
    rows.append(pd.DataFrame({"text": ds["text"].astype(str), "label": ds["label"].astype(int),
                              "source": SRC_DS}))
    pr = pd.read_csv(raw_path(SRC_PR, "prompt_injections.csv"))
    rows.append(pd.DataFrame({"text": pr["text"].astype(str), "label": 1, "source": SRC_PR,
                              "category": pr["category"].astype(str),
                              "language": pr["language"].astype(str)}))
    df = pd.concat(rows, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    for c in ("category", "language"):
        if c not in df:
            df[c] = None
    return df


def clean_pool(df: pd.DataFrame, report: dict = None) -> pd.DataFrame:
    """Dedup + conflict removal (deterministic, seed-independent)."""
    rep = report if report is not None else {}
    df = df[df.text.str.len() > 5].copy()
    df["key"] = df.text.map(norm_key)

    # (2) label conflicts: same text with both labels -> keep as positive only
    lab = df.groupby("key").label.agg(["min", "max"])
    conflict = set(lab[(lab["min"] == 0) & (lab["max"] == 1)].index)
    rep["label_conflicts_resolved_to_positive"] = len(conflict)
    df = df[~(df.key.isin(conflict) & (df.label == 0))]

    # (1) cross-source exact dedupe; source priority jailbreak_llms > deepset > pr1m8
    prio = {SRC_JB: 0, SRC_DS: 1, SRC_PR: 2}
    df["prio"] = df.source.map(prio)
    df = df.sort_values(["prio"], kind="stable")
    first_source = df.groupby("key").source.first()
    all_sources = df.groupby("key").source.agg(lambda s: "|".join(sorted(set(s))))
    # remember pr1m8 metadata even when the surviving row came from deepset
    pr_meta = df[df.source == SRC_PR].drop_duplicates("key").set_index("key")[["category", "language"]]
    rep["pr1m8_rows_raw"] = int((df.source == SRC_PR).sum())
    rep["pr1m8_rows_duplicating_other_source"] = int(
        ((df.source == SRC_PR) & (df.key.map(first_source) != SRC_PR)).sum())
    before = len(df)
    df = df.drop_duplicates("key", keep="first").copy()
    rep["exact_duplicates_removed"] = before - len(df)
    df["all_sources"] = df.key.map(all_sources)
    df["dup_of_other_source"] = df.all_sources.str.contains(r"\|")
    df["pr1m8_category"] = df.key.map(pr_meta["category"])
    df["language"] = df.key.map(pr_meta["language"]).where(df.key.isin(pr_meta.index), df["language"])
    return df.drop(columns=["prio"]).reset_index(drop=True)


def load_pool(seed: int, neg_ratio: int = 3, report: dict = None) -> pd.DataFrame:
    """Cleaned pool with benign capped at neg_ratio x positives (seeded)."""
    rep = report if report is not None else {}
    df = clean_pool(load_raw_rows(), rep)
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    n_neg = min(len(neg), neg_ratio * len(pos))
    neg = neg.sample(n=n_neg, random_state=seed)
    pool = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rep.update({"n_total": len(pool), "n_pos": int(pool.label.sum()),
                "n_neg": int((pool.label == 0).sum()),
                "by_source": pool.groupby(["source", "label"]).size().to_dict()})
    return pool


def fingerprint(df: pd.DataFrame) -> str:
    """Short hash of (text,label) content - printed by every script so the
    paper can show all experiments used the same data."""
    h = hashlib.sha256()
    for k, y in sorted(zip(df.text.map(norm_key), df.label.astype(int))):
        h.update(k.encode("utf-8"))
        h.update(str(y).encode())
    return h.hexdigest()[:12]


# --------------------------------------------------------- near duplicates ---
def near_duplicate_groups(texts, threshold: float = 0.8, chunk: int = 1000) -> np.ndarray:
    """Group ids such that any two texts with TF-IDF (word 1-3gram, binary)
    cosine >= threshold end up in the same group (union-find)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 3), binary=True, min_df=1, sublinear_tf=False,
                          max_features=200000)
    X = vec.fit_transform([norm_key(t) for t in texts])
    n = X.shape[0]
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, chunk):
        S = (X[start:start + chunk] @ X.T).toarray()
        rows, cols = np.nonzero(S >= threshold)
        for r, c in zip(rows, cols):
            a, b = find(start + r), find(c)
            if a != b:
                parent[a] = b
    return np.array([find(i) for i in range(n)])


# ------------------------------------------------------------------ splits ---
def make_split(pool: pd.DataFrame, seed: int, mode: str = "random",
               fracs=(0.70, 0.15, 0.15), groups: np.ndarray = None):
    """Stratified train/val/test split.
    mode="random":  per-class shuffle and cut (the original protocol)
    mode="grouped": near-duplicate groups never straddle splits"""
    rng = np.random.default_rng(seed)
    assign = np.empty(len(pool), dtype=object)
    if mode == "random":
        for y in (0, 1):
            idx = np.where(pool.label.values == y)[0]
            idx = idx[rng.permutation(len(idx))]
            n_tr, n_va = int(len(idx) * fracs[0]), int(len(idx) * fracs[1])
            assign[idx[:n_tr]] = "train"
            assign[idx[n_tr:n_tr + n_va]] = "val"
            assign[idx[n_tr + n_va:]] = "test"
    elif mode == "grouped":
        if groups is None:
            groups = near_duplicate_groups(pool.text.tolist())
        g = pd.DataFrame({"g": groups, "y": pool.label.values})
        # a group's stratum = majority label; fill splits group by group
        gl = g.groupby("g").y.agg(["mean", "size"])
        for y in (0, 1):
            gids = gl[(gl["mean"] >= 0.5) == bool(y)].index.values
            gids = gids[rng.permutation(len(gids))]
            sizes = gl.loc[gids, "size"].values
            total, acc = sizes.sum(), 0
            for gid, sz in zip(gids, sizes):
                frac = acc / total
                split = "train" if frac < fracs[0] else ("val" if frac < fracs[0] + fracs[1] else "test")
                assign[np.where(groups == gid)[0]] = split
                acc += sz
    else:
        raise ValueError(mode)
    out = {}
    for s in ("train", "val", "test"):
        d = pool[assign == s]
        out[s] = d.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out["train"], out["val"], out["test"]


def describe(train, val, test) -> str:
    def f(d):
        return f"{len(d)} ({int(d.label.sum())} pos / {int((d.label == 0).sum())} neg)"
    return f"train {f(train)} | val {f(val)} | test {f(test)}"


def save_manifest(extra: dict = None):
    m = {k: {"repo": v["repo"], "commit": v["sha"], "files": v["files"]} for k, v in PINNED.items()}
    if extra:
        m.update(extra)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "MANIFEST.json").write_text(json.dumps(m, indent=2))
