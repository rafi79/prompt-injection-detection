"""
B3: meta-llama/Prompt-Guard-86M behind the common Detector interface.

finetune=True  : replace the 3-class head with a fresh 2-class head and train
                 ALL parameters (not only the head) - 3 epochs, batch 16,
                 lr 2e-5, linear warm-up 10%, max_len 256, class-weighted
                 cross-entropy, keep the best-validation-F1 epoch, then tune
                 the decision threshold on validation like every system.
finetune=False : out-of-the-box model, score = 1 - P(BENIGN).

Needs: torch, transformers, a Hugging Face login with the model licence
accepted (gated). Set PI_PROMPTGUARD_ID to use another local/remote path.
"""
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.systems import Detector, tune_threshold

MODEL_ID = os.environ.get("PI_PROMPTGUARD_ID", "meta-llama/Prompt-Guard-86M")
MAX_LEN, BATCH_SIZE, EPOCHS, LR = 256, 16, 3, 2e-5
# Each batch of 16 is processed as micro-batches of MICRO_BATCH with gradient
# accumulation. The loss is normalised by the class-weight sum of the FULL
# batch, so the gradient is identical to a single batch of 16; only peak GPU
# memory drops (needed on a 6 GB laptop GPU, where Windows otherwise spills
# into much slower shared system memory).
MICRO_BATCH = int(os.environ.get("PI_B3_MICRO_BATCH", "4"))


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PromptGuardDetector(Detector):
    def __init__(self, finetune=True, seed=42, device=None, epochs=EPOCHS, verbose=True):
        self.name = "B3" if finetune else "B3raw"
        self.finetune = finetune
        self.seed = seed
        self.epochs = epochs
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------- utils
    def _collate(self, batch):
        texts = [b[0] for b in batch]
        enc = self.tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        if batch[0][1] is not None:
            enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return enc

    def _probs_from_logits(self, logits):
        p = torch.softmax(logits.float(), dim=-1)
        if self.finetune:
            return p[:, 1]
        return 1.0 - p[:, self.benign_idx]

    @torch.no_grad()
    def scores(self, texts, batch_size=16):
        self.model.eval()
        out = []
        data = [(t, None) for t in texts]
        for enc in DataLoader(data, batch_size=batch_size, collate_fn=self._collate):
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out.append(self._probs_from_logits(self.model(**enc).logits).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    @torch.no_grad()
    def score_one(self, text):
        self.model.eval()
        enc = self.tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.device)
        p = self._probs_from_logits(self.model(**enc).logits)
        val = float(p[0].item())  # .item() synchronises the GPU
        return val

    # ------------------------------------------------------------------ fit
    def fit(self, tr_x, tr_y, va_x, va_y):
        set_all_seeds(self.seed)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        if not self.finetune:
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(self.device)
            labels = {v.upper(): int(k) for k, v in self.model.config.id2label.items()}
            self.benign_idx = labels.get("BENIGN", 0)
            return self._tune(va_x, va_y)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=2, ignore_mismatched_sizes=True).to(self.device)
        tr_y = np.asarray(tr_y, dtype=int)
        g = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(list(zip(list(tr_x), tr_y.tolist())), batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=self._collate, generator=g)
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=tr_y)
        wt = torch.tensor(w, dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=wt, reduction="sum")
        opt = torch.optim.AdamW(self.model.parameters(), lr=LR)   # ALL parameters
        total = len(loader) * self.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

        best_f1, best_state = -1.0, None
        for ep in range(self.epochs):
            self.model.train()
            run = 0.0
            for step, enc in enumerate(loader):
                labels_all = enc.pop("labels").to(self.device)
                denom = wt[labels_all].sum()          # weighted-mean denominator of the full batch
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for i in range(0, len(labels_all), MICRO_BATCH):
                    mb = {k: v[i:i + MICRO_BATCH].to(self.device) for k, v in enc.items()}
                    loss = loss_fn(self.model(**mb).logits, labels_all[i:i + MICRO_BATCH]) / denom
                    loss.backward()
                    batch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                run += batch_loss
                if self.verbose and (step + 1) % 50 == 0:
                    print(f"      step {step + 1}/{len(loader)}", flush=True)
            va_p = self.scores(va_x)
            f1 = f1_score(np.asarray(va_y), (va_p >= 0.5).astype(int), zero_division=0)
            if self.verbose:
                print(f"    [B3 seed {self.seed}] epoch {ep + 1}/{self.epochs} "
                      f"loss={run / max(len(loader), 1):.4f} val_F1@0.5={f1:.4f}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        del opt, sched, best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self._tune(va_x, va_y)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        with open(os.path.join(path, "threshold.txt"), "w") as f:
            f.write(str(self.threshold))

    @classmethod
    def load(cls, path, device=None):
        d = cls(finetune=True, device=device)
        d.tok = AutoTokenizer.from_pretrained(path)
        d.model = AutoModelForSequenceClassification.from_pretrained(path).to(d.device)
        with open(os.path.join(path, "threshold.txt")) as f:
            d.threshold = float(f.read().strip())
        return d
