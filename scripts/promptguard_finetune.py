"""
Fine-tunes meta-llama/Prompt-Guard-86M on our own training distribution,
as Meta's own model card recommends for real use ("As a fine-tuned solution
... to achieve very high precision and recall of malicious application
specific prompts"). This replaces the out-of-the-box 3-class head
(BENIGN/INJECTION/JAILBREAK, which saturates to ~100% non-benign on our
distribution - see scripts/promptguard_calibrated.py) with a 2-class head
matching our binary label space, and fine-tunes it end-to-end.

Runs comfortably on a 6GB GPU: 86M parameters, batch size 16, ~4.6K train
examples, 3 epochs takes a few minutes.

SETUP: same as scripts/promptguard_baseline.py (HF login + license accept
required for this gated model).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODEL_ID = "meta-llama/Prompt-Guard-86M"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5


class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(batch["label"].numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def main():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    test = pd.read_csv(DATA / "test.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_ID} with a fresh 2-class head (discarding the "
          f"pretrained 3-class BENIGN/INJECTION/JAILBREAK head)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    train_ds = PromptDataset(train.text.tolist(), train.label.tolist(), tokenizer)
    val_ds = PromptDataset(val.text.tolist(), val.label.tolist(), tokenizer)
    test_ds = PromptDataset(test.text.tolist(), test.label.tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train.label.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights (balanced): benign={class_weights[0]:.3f}, attack={class_weights[1]:.3f}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                 num_training_steps=total_steps)

    best_val_f1 = -1
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(train_loader)} loss={loss.item():.4f}")

        val_probs, val_labels = evaluate(model, val_loader, device)
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}  val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val F1={best_val_f1:.4f}, restoring best checkpoint.")
    model.load_state_dict(best_state)

    # calibrate threshold on validation set (same protocol as our own scorer)
    val_probs, val_labels = evaluate(model, val_loader, device)
    best_t, best_f1_t = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1_t:
            best_f1_t, best_t = f1, t
    print(f"Calibrated threshold={best_t:.2f} (val F1={best_f1_t:.4f})")

    # evaluate on test set with metrics
    test_probs, test_labels = evaluate(model, test_loader, device)
    test_preds = (test_probs >= best_t).astype(int)

    # single-request latency, matching the methodology used for every other system
    model.eval()
    lats = []
    with torch.no_grad():
        for t in test.text.tolist():
            t0 = time.perf_counter()
            enc = tokenizer(t, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            _ = model(**enc).logits
            lats.append((time.perf_counter() - t0) * 1000.0)

    metrics = {
        "system": f"B3: PromptGuard-86M fine-tuned ({device})",
        "accuracy": accuracy_score(test_labels, test_preds),
        "precision": precision_score(test_labels, test_preds, zero_division=0),
        "recall": recall_score(test_labels, test_preds, zero_division=0),
        "f1": f1_score(test_labels, test_preds, zero_division=0),
        "roc_auc": roc_auc_score(test_labels, test_probs),
        "latency_ms_mean": float(np.mean(lats)),
        "latency_ms_p50": float(np.percentile(lats, 50)),
        "latency_ms_p95": float(np.percentile(lats, 95)),
        "n_test": len(test),
    }
    print("\n" + pd.Series(metrics).to_string())

    out = RESULTS / "promptguard_finetuned_results.csv"
    pd.DataFrame([metrics]).to_csv(out, index=False)

    save_dir = RESULTS / "promptguard_finetuned_model"
    save_dir.mkdir(exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nSaved metrics to {out}")
    print(f"Saved fine-tuned model to {save_dir}")


if __name__ == "__main__":
    main()
