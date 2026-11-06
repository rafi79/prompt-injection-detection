# Semantic-Lite Prompt Injection Detection

Code and experiments for the paper *"Semantic-Lite Prompt Injection Detection
for Real-Time Web-Based AI Assistants"* (revision after review).

Every number in the paper is produced by a script in `scripts/` and stored as
a CSV in `results/`. All experiments use **one dataset version**, rebuilt
deterministically from raw files pinned to exact upstream commits.

## Data (pinned)

| Source | Upstream repo @ commit | Files |
|---|---|---|
| JailbreakHub (Shen et al., CCS'24) | `verazuo/jailbreak_llms` @ `4f4031bf8be187f4478c7f94f42b08714722c12e` | `data/prompts/{jailbreak,regular}_prompts_2023_12_25.csv` |
| deepset prompt-injections (parquet mirror) | `sinanw/llm-security-prompt-injection` @ `c8af0b015a9a4f3380f9e20214828c2dbeadc72c` | `data/raw/{train,test}-*.parquet` |
| pr1m8 categorized injections | `pr1m8/prompt-injections` @ `a0ee668c48793f40350489e34727a47d24eeb21f` | `prompt_injections.csv` |

`scripts/fetch_data.py` downloads exactly these files from
raw.githubusercontent.com and checks their SHA-256 hashes (listed in
`src/data.py` and written to `data/raw/MANIFEST.json`).

Cleaning (`src/data.py`): whitespace-normalised, case-folded exact
de-duplication across sources; texts listed with both labels kept as
positives only; texts of <= 5 characters dropped; benign capped at 3x
positives. Findings from the audit:

* 51 of pr1m8's 82 prompts are verbatim copies of deepset prompts. They are
  kept once (attributed to deepset); held-out pr1m8 evaluation uses only the
  31 prompts that occur nowhere else.
* 42 JailbreakHub prompts appear in both the jailbreak and the regular file.
* 1,174 texts fall into near-duplicate groups (TF-IDF cosine >= 0.8); 53 of
  those groups contain both labels. `run_main.py` therefore also reports a
  group-aware split in which no near-duplicate group straddles train/val/test.

Resulting pool (seed 42): 6,600 prompts (1,650 injection / 4,950 benign),
data fingerprint `bd868eb585b4` (printed by every script).

## Systems (`src/systems.py`)

| Name | Description |
|---|---|
| B1 | keyword/pattern engine only (flag if >= 1 hit) |
| v1 | lexical risk scorer: 11 learned lexical/structural features + logistic regression |
| v2 | v1 + semantic-lite (word LSA, class centroids; original, in-sample features) |
| v2cf | v2 with cross-fitted (out-of-fold) semantic features |
| v3 | v1 + semantic-lite (word+char LSA, 8 prototypes/class, kNN-10, cross-fitted) |
| B2 | TF-IDF (word 1-2 grams, 20k) + calibrated linear SVM |
| B3 / B3raw | PromptGuard-86M fine-tuned (ALL parameters) / out of the box |
| LenOnly | length-only logistic regression (shortcut diagnostic) |

All learned systems choose their decision threshold on the validation split
(max F1, grid 0.02-0.98). Sanitizer (`src/sanitizer.py`) has three modes:
`none`, `legacy` (exact original behaviour) and `full` (new default).

## Reproduce (Windows PowerShell, from the repo root, `.venv` active)

```powershell
$env:PYTHONUTF8 = "1"
py -m pip install -r requirements.txt
py scripts\fetch_data.py
py scripts\build_dataset.py

# 1. main multi-seed table, random + near-duplicate-grouped splits (CPU)
py scripts\run_main.py
# 2. add PromptGuard (GPU) to the same tables
py scripts\run_main.py --systems B3 B3raw --modes random
py scripts\run_main.py --systems B3 --modes grouped
# 3. leave-one-source-out, all systems
py scripts\run_loso.py
py scripts\run_loso.py --systems B3
# 4. obfuscation suite + benign controls
py scripts\run_obfuscation.py --with-b3
# 5. one latency / throughput / memory harness
py scripts\run_latency.py --with-b3
# 6. paraphrase robustness (generate on GPU, annotate 60 rows, evaluate)
py scripts\run_paraphrase.py generate
#    -> fill results\paraphrase_validity_sheet.csv (intent_preserved = 1 or 0)
py scripts\run_paraphrase.py evaluate --with-b3
# 7. adaptive black-box attack with query budgets (+ windowed mitigation)
py scripts\run_adaptive.py --with-b3
#    -> fill results\adaptive_validity_sheet.csv, then
py scripts\score_validity.py results\adaptive_validity_sheet.csv
# 8. universal evasion suffix against the detectors
py scripts\run_universal.py --with-b3 --window
# 9. multilingual (NLLB-200 translation on GPU, then evaluate)
py scripts\run_multilingual.py translate
py scripts\run_multilingual.py evaluate --with-b3
```

Steps 4-9 load the seed-42 models that step 1-2 save in `results/models/`
(git-ignored). B3 needs a Hugging Face login with the Prompt-Guard-86M
licence accepted.

## Layout

* `src/` - sanitizer, keyword engine, features, semantic-lite encoder,
  systems, PromptGuard wrapper, data pipeline, statistics, attack operators
* `scripts/` - experiment entry points (`scripts/legacy/` holds the
  superseded first-submission scripts, kept for history only)
* `results/` - CSV outputs; `results/preds/` per-example predictions used
  for the paired significance tests; `results/legacy_v1/` the
  first-submission outputs (different data version - not used in the paper)
