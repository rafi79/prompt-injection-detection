"""
B3 baseline: Meta's PromptGuard-86M, a real fine-tuned transformer classifier
for prompt injection / jailbreak detection. This requires internet access to
Hugging Face Hub (blocked in the sandbox this project was originally built
in) and works fine on your 6GB GPU or even CPU (it's an 86M-parameter model,
~350MB).

SETUP (run once):
    pip install transformers torch --index-url https://download.pytorch.org/whl/cu121
    # (use the plain `pip install torch` command from pytorch.org if you don't have
    #  an NVIDIA GPU set up with CUDA; the model still runs fine on CPU)
    pip install accelerate

NOTE: meta-llama/Prompt-Guard-86M is a gated model on Hugging Face. You will
need to:
    1. Create a free Hugging Face account at https://huggingface.co/join
    2. Accept the license at https://huggingface.co/meta-llama/Prompt-Guard-86M
    3. Run `huggingface-cli login` and paste an access token from
       https://huggingface.co/settings/tokens
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

MODEL_ID = "meta-llama/Prompt-Guard-86M"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def predict_one(text, tokenizer, model, device):
    t0 = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # PromptGuard label space: 0=benign, 1=injection, 2=jailbreak (check model card;
    # this script treats any non-benign class as a positive prediction)
    pred_class = int(torch.argmax(probs).item())
    label = 1 if pred_class != 0 else 0
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return label, latency_ms


def main():
    test = pd.read_csv(DATA / "test.csv")
    tokenizer, model, device = load_model()

    preds, lats = [], []
    for i, text in enumerate(test.text.tolist()):
        label, lat = predict_one(text, tokenizer, model, device)
        preds.append(label)
        lats.append(lat)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test)}")

    preds = np.array(preds)
    labels = test.label.values

    metrics = {
        "system": f"B3: PromptGuard-86M ({device})",
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "latency_ms_mean": float(np.mean(lats)),
        "latency_ms_p50": float(np.percentile(lats, 50)),
        "latency_ms_p95": float(np.percentile(lats, 95)),
        "n_test": len(test),
    }
    print("\n" + pd.Series(metrics).to_string())

    out = RESULTS / "promptguard_baseline_results.csv"
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
