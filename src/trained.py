"""Load the seed-42 / random-split models saved by scripts/run_main.py, so
that obfuscation, paraphrase, adaptive-attack and latency experiments all
use exactly the detectors reported in the main table."""
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "results" / "models"


def load_trained(name):
    if name == "B3":
        from src.promptguard import PromptGuardDetector
        path = MODELS / "b3_seed42"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: py scripts/run_main.py --systems B3 --modes random --seeds 42")
        return PromptGuardDetector.load(str(path))
    path = MODELS / f"{name}_seed42.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/run_main.py first")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_many(names):
    out = {}
    for n in names:
        try:
            out[n] = load_trained(n)
        except FileNotFoundError as e:
            print(f"  [skip {n}] {e}")
    return out
