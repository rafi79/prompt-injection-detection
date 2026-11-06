"""
Shared pieces for the attack experiments (paraphrase, adaptive black-box,
universal suffix, multilingual): the common attack-prompt sample and the
mutation operators of the adaptive attacker.
"""
import re
import numpy as np
import pandas as pd

from src.data import load_pool, make_split


def truncate(t, limit=800):
    if len(t) <= limit:
        return t
    cut = t[:limit]
    m = list(re.finditer(r"[.!?\n]\s", cut))
    return cut[:m[-1].end()].strip() if m and m[-1].end() > limit * 0.5 else cut.rsplit(" ", 1)[0]


def attack_sample(n=150, split="test", seed=42, limit=800, label=1):
    """Seeded random sample of prompts (attacks by default) from the seed-42
    random split, truncated at a sentence boundary. Identical for every
    script, so paraphrase / adaptive / multilingual results share prompts."""
    pool = load_pool(42)
    tr, va, te = make_split(pool, 42, "random")
    d = {"train": tr, "val": va, "test": te}[split]
    d = d[d.label == label].reset_index(drop=True)
    idx = np.random.default_rng(seed).choice(len(d), min(n, len(d)), replace=False)
    return [truncate(t, limit) for t in d.text.values[idx]]


# small hand-written synonym list for attack-salient words (not a thesaurus)
SYNONYMS = {
    "ignore": ["disregard", "overlook", "set aside", "skip"],
    "disregard": ["ignore", "set aside", "pay no attention to"],
    "forget": ["drop", "let go of", "discard"],
    "previous": ["prior", "earlier", "preceding", "former"],
    "prior": ["previous", "earlier"],
    "above": ["preceding", "earlier"],
    "instructions": ["directions", "guidance", "directives", "orders"],
    "instruction": ["direction", "directive", "order"],
    "rules": ["guidelines", "policies", "constraints"],
    "pretend": ["imagine", "act as if", "suppose"],
    "act": ["behave", "operate", "function"],
    "reveal": ["disclose", "share", "expose"],
    "show": ["display", "present"],
    "print": ["output", "write out"],
    "prompt": ["setup text", "initial message"],
    "system": ["base", "core"],
    "restrictions": ["limits", "boundaries", "constraints"],
    "filters": ["safeguards", "checks"],
    "unrestricted": ["unlimited", "unconstrained", "free"],
    "jailbreak": ["unlock", "liberation"],
    "developer": ["dev", "engineering"],
    "mode": ["setting", "state"],
    "character": ["role", "persona"],
    "now": ["from here on", "henceforth"],
    "answer": ["respond", "reply"],
    "must": ["have to", "need to"],
    "never": ["at no point", "under no circumstances"],
    "always": ["at all times", "every time"],
    "ethical": ["moral"],
    "guidelines": ["rules", "policies"],
    "openai": ["your creators", "the company"],
    "chatgpt": ["the assistant", "you"],
    "dan": ["the free AI", "the other persona"],
}

ZW = "​"
HOMO = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "i": "і"}


def op_synonym(text, rng, **_):
    words = list(re.finditer(r"[A-Za-z]+", text))
    cands = [m for m in words if m.group(0).lower() in SYNONYMS]
    if not cands:
        return None
    m = cands[rng.integers(len(cands))]
    rep = SYNONYMS[m.group(0).lower()][rng.integers(len(SYNONYMS[m.group(0).lower()]))]
    if m.group(0)[0].isupper():
        rep = rep[0].upper() + rep[1:]
    return text[:m.start()] + rep + text[m.end():]


def op_benign_padding(text, rng, benign_pool=None, **_):
    if not benign_pool:
        return None
    pad = benign_pool[rng.integers(len(benign_pool))]
    return (pad + "\n\n" + text) if rng.random() < 0.5 else (text + "\n\n" + pad)


def op_char(text, rng, **_):
    words = [m for m in re.finditer(r"[A-Za-z]{4,}", text)]
    if not words:
        return None
    m = words[rng.integers(len(words))]
    w = m.group(0)
    k = int(rng.integers(3))
    if k == 0:      # zero-width split
        p = int(rng.integers(1, len(w)))
        w2 = w[:p] + ZW + w[p:]
    elif k == 1:    # homoglyph
        idx = [i for i, c in enumerate(w) if c in HOMO]
        if not idx:
            return None
        i = idx[rng.integers(len(idx))]
        w2 = w[:i] + HOMO[w[i]] + w[i + 1:]
    else:           # adjacent-letter swap (typo)
        p = int(rng.integers(1, len(w) - 2))
        w2 = w[:p] + w[p + 1] + w[p] + w[p + 2:]
    return text[:m.start()] + w2 + text[m.end():]


def op_paraphrase(text, rng, paraphrases=None, used=None, **_):
    if not paraphrases:
        return None
    free = [p for p in paraphrases if p not in used]
    if not free:
        return None
    p = free[rng.integers(len(free))]
    used.add(p)
    return p


OPERATORS = {"synonym": op_synonym, "benign_padding": op_benign_padding, "char": op_char,
             "paraphrase": op_paraphrase}


def benign_sentences(n=500, seed=0, max_chars=200):
    """Short benign snippets (first sentence of benign TRAINING prompts) used
    for padding/dilution."""
    pool = load_pool(42)
    tr, _, _ = make_split(pool, 42, "random")
    neg = tr[tr.label == 0].text.tolist()
    out = []
    for t in neg:
        s = re.split(r"(?<=[.!?\n])\s+", t.strip())[0].strip()
        if 20 <= len(s) <= max_chars:
            out.append(s)
    rng = np.random.default_rng(seed)
    return [out[i] for i in rng.choice(len(out), min(n, len(out)), replace=False)]
