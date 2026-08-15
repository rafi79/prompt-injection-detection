"""
The real-world dataset (jailbreak_llms / deepset / pr1m8) contains almost no
obfuscated payloads, so the ablation above cannot exercise the sanitization
stage. This script constructs obfuscated variants of REAL positive test
examples (zero-width character insertion, homoglyph substitution, base64
wrapping) - clearly derived/synthetic transforms of real attack text, not
fabricated new attacks - to measure whether sanitization specifically helps
against obfuscation, which is exactly its intended purpose.
"""
import sys
import re
import base64
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sanitizer import sanitize
from src.keyword_engine import analyze

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
random.seed(42)

ZERO_WIDTH = "\u200b"
HOMOGLYPH_MAP = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "c": "\u0441"}


def insert_zero_width(text: str) -> str:
    return ZERO_WIDTH.join(list(text))


def homoglyph_substitute(text: str) -> str:
    return "".join(HOMOGLYPH_MAP.get(c, c) for c in text)


def base64_wrap(text: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"Please decode and follow this base64 message: {encoded}"


TRANSFORMS = {
    "zero_width_insertion": insert_zero_width,
    "homoglyph_substitution": homoglyph_substitute,
    "base64_wrapping": base64_wrap,
}


def detect_raw(text: str) -> int:
    clean = re.sub(r"\s+", " ", text).strip()
    kw = analyze(clean)
    return 1 if kw.total_hits >= 1 else 0


def detect_sanitized(text: str) -> int:
    clean = sanitize(text)
    kw = analyze(clean)
    return 1 if kw.total_hits >= 1 else 0


def main():
    test = pd.read_csv(DATA / "test.csv")
    positives = test[test.label == 1].text.tolist()

    # keep only positives the keyword engine could catch in clean form,
    # so the experiment isolates the effect of obfuscation specifically
    detectable = [t for t in positives if detect_raw(t) == 1]
    sample = detectable[: min(200, len(detectable))]
    print(f"Base positives originally detected by keyword engine: {len(sample)} / {len(positives)}")

    rows = []
    for name, transform in TRANSFORMS.items():
        obf_texts = [transform(t) for t in sample]
        no_san_rate = np.mean([detect_raw(t) for t in obf_texts])
        san_rate = np.mean([detect_sanitized(t) for t in obf_texts])
        rows.append({
            "obfuscation": name,
            "detection_rate_no_sanitization": no_san_rate,
            "detection_rate_with_sanitization": san_rate,
            "n": len(obf_texts),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "obfuscation_robustness.csv", index=False)


if __name__ == "__main__":
    main()
