# Lightweight Prompt Injection Detection

Real, runnable implementation and evaluation for the paper *"Lightweight
Prompt Injection Detection for Web-Based AI Assistant Applications."*

## Data sources (all real, all cited)

| Source | Content | Used as |
|---|---|---|
| [verazuo/jailbreak_llms](https://github.com/verazuo/jailbreak_llms) (ACM CCS'24) | 1,405 real in-the-wild jailbreak prompts + 13,735 real regular ChatGPT prompts | positives + negatives |
| deepset prompt-injections benchmark (mirrored via sinanw/llm-security-prompt-injection) | 546 train + 116 test labeled injection/benign prompts | positives + negatives |
| [pr1m8/prompt-injections](https://github.com/pr1m8/prompt-injections) | 82 categorized direct injection examples | positives |

Combined, deduplicated, benign capped at 3x positives to avoid extreme
imbalance: **6,636 examples** (1,659 injection / 4,977 benign), split
70/15/15 train/val/test. No synthetic or fabricated text was added to this
core dataset — see `scripts/build_dataset.py`.

## Pipeline

`src/sanitizer.py` → `src/keyword_engine.py` → `src/risk_scorer.py`,
orchestrated by `src/pipeline.py`. See the paper for the architecture
diagram.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/build_dataset.py       # builds data/processed/{train,val,test}.csv
python scripts/train.py               # trains risk scorer + TF-IDF/SVM baseline
python scripts/benchmark.py           # accuracy/precision/recall/F1 + latency + memory
python scripts/ablation.py            # stage-by-stage ablation
python scripts/obfuscation_robustness.py  # sanitization-specific robustness test
```

## Honest scope notes

- **B2 baseline is TF-IDF + Linear SVM**, not a fine-tuned transformer.
  This sandbox has no network access to Hugging Face Hub, so a real
  PromptGuard/DistilBERT-style baseline could not be downloaded or run
  here. If you have GPU/network access, swap in `transformers` +
  a pretrained checkpoint using the same `predict_label_timed` interface
  in `src/baselines.py`.
- **No LLM-judge (B3) baseline was run** — no API credentials are
  configured in this environment. `src/pipeline.py`'s interface is
  compatible with adding one (call an LLM API, time it, compare).
- All reported numbers in `results/*.csv` are from actual runs, not
  placeholders.
