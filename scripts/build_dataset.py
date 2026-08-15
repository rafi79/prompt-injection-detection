"""
Build a unified labeled dataset (label=1 injection/jailbreak, label=0 benign)
from three real, publicly available sources:

  1. verazuo/jailbreak_llms (ACM CCS'24) - 1,405 real in-the-wild jailbreak
     prompts + 13,735 real regular ChatGPT prompts.
  2. deepset/prompt-injections benchmark (mirrored locally as parquet by
     sinanw/llm-security-prompt-injection) - 546 train + 116 test labeled
     prompt-injection examples, a dataset used across the prompt-injection
     defense literature.
  3. pr1m8/prompt-injections - 82 categorized direct prompt-injection
     examples (instruction override, role-play, priority manipulation, etc.)

No synthetic or fabricated examples are included. Class balance is controlled
by subsampling the (much larger) benign pool rather than inventing data.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)


def load_jailbreak_llms():
    jb = pd.read_csv(RAW / "jailbreak_llms_prompts" / "jailbreak_prompts_2023_12_25.csv")
    reg = pd.read_csv(RAW / "jailbreak_llms_prompts" / "regular_prompts_2023_12_25.csv")
    jb_df = pd.DataFrame({"text": jb["prompt"].astype(str), "label": 1, "source": "jailbreak_llms_ccs24"})
    reg_df = pd.DataFrame({"text": reg["prompt"].astype(str), "label": 0, "source": "jailbreak_llms_ccs24"})
    return jb_df, reg_df


def load_deepset():
    tr = pd.read_parquet(RAW / "deepset_train.parquet")
    te = pd.read_parquet(RAW / "deepset_test.parquet")
    df = pd.concat([tr, te], ignore_index=True)
    df = df.rename(columns={"text": "text", "label": "label"})
    df["source"] = "deepset_prompt_injections"
    return df[df.label == 1][["text", "label", "source"]], df[df.label == 0][["text", "label", "source"]]


def load_pr1m8():
    df = pd.read_csv(RAW / "pr1m8_prompt_injections.csv")
    out = pd.DataFrame({"text": df["text"].astype(str), "label": 1, "source": "pr1m8_prompt_injections"})
    return out, pd.DataFrame(columns=["text", "label", "source"])


def main():
    jb_pos, jb_neg = load_jailbreak_llms()
    ds_pos, ds_neg = load_deepset()
    pr_pos, pr_neg = load_pr1m8()

    positives = pd.concat([jb_pos, ds_pos, pr_pos], ignore_index=True)
    negatives = pd.concat([jb_neg, ds_neg, pr_neg], ignore_index=True)

    # basic cleanup: drop empty/very short/duplicate text
    for df in (positives, negatives):
        df["text"] = df["text"].astype(str).str.strip()
    positives = positives[positives.text.str.len() > 5].drop_duplicates(subset="text")
    negatives = negatives[negatives.text.str.len() > 5].drop_duplicates(subset="text")

    # cap benign at 3x positive count to avoid extreme imbalance while
    # staying realistic (benign traffic dominates in production).
    n_pos = len(positives)
    n_neg_target = min(len(negatives), n_pos * 3)
    negatives = negatives.sample(n=n_neg_target, random_state=SEED).reset_index(drop=True)

    data = pd.concat([positives, negatives], ignore_index=True)
    data = data.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # stratified 70/15/15 split
    def split(df):
        n = len(df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]

    pos = data[data.label == 1].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    neg = data[data.label == 0].sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    pos_tr, pos_va, pos_te = split(pos)
    neg_tr, neg_va, neg_te = split(neg)

    train = pd.concat([pos_tr, neg_tr]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val = pd.concat([pos_va, neg_va]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test = pd.concat([pos_te, neg_te]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train.to_csv(OUT / "train.csv", index=False)
    val.to_csv(OUT / "val.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)

    print("Dataset sources:")
    print(data["source"].value_counts())
    print("\nOverall label balance:", data["label"].value_counts().to_dict())
    print(f"\nTrain: {len(train)} ({train.label.sum()} pos / {len(train)-train.label.sum()} neg)")
    print(f"Val:   {len(val)} ({val.label.sum()} pos / {len(val)-val.label.sum()} neg)")
    print(f"Test:  {len(test)} ({test.label.sum()} pos / {len(test)-test.label.sum()} neg)")


if __name__ == "__main__":
    main()
