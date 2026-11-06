"""
Stage 1: Sanitization.

Normalizes raw text before analysis so that later stages see a canonical
form rather than an obfuscated one:
  - Unicode NFKC normalization (also folds full-width characters)
  - zero-width / invisible character stripping
  - homoglyph folding for commonly abused Cyrillic/Greek lookalikes
  - HTML/markdown tag and comment stripping (indirect web content)
  - decoding of encoded payloads: base64 (standard or URL-safe, padded or
    not, >= 16 chars), hex (plain, \\x-escaped or space-separated), applied
    recursively up to depth 3
  - de-obfuscation views: ROT13, reversed text, leetspeak, and
    spaced/dotted letters ("i g n o r e", "i.g.n.o.r.e")

Decoded / de-obfuscated views are APPENDED after a "[DECODED]" marker; the
original text is never discarded, so the downstream stages still see the
(suspicious) encoded form as well.

`sanitize(text, mode="none")` only collapses whitespace (no sanitization);
`sanitize(text, mode="legacy")` reproduces the original (paper v1) behaviour
exactly, so the old and new sanitizers can be compared in the same run.

Root cause of the original 37.6% base64 recovery (found 2026-10): the legacy
decoder rejected every decoded payload for which `str.isprintable()` was
False. `isprintable()` is False for "\\n" and "\\t", so any multi-line attack
prompt was silently dropped after a successful decode. The 40-character
minimum was a secondary issue (it only matters for short payloads).
"""
import re
import base64
import binascii
import codecs
import unicodedata
from bs4 import BeautifulSoup

# ---------------------------------------------------------------- legacy ---
_LEGACY_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_LEGACY_HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "і": "i", "ԁ": "d", "һ": "h", "ѕ": "s",
}
_LEGACY_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

# ------------------------------------------------------------------ full ---
ZERO_WIDTH = re.compile(
    r"[­͏؜ᅟᅠ឴឵᠎​-‏"
    r"‪-‮⁠-⁤⁦-⁯ㅤ︀-️﻿ﾠ]"
)

HOMOGLYPHS = {
    # Cyrillic lower
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "ԁ": "d", "һ": "h", "к": "k", "н": "h", "т": "t",
    "м": "m", "в": "b", "ґ": "r",
    # Cyrillic upper
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    # Greek
    "α": "a", "ο": "o", "ε": "e", "ι": "i", "κ": "k",
    "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}

# base64 candidates: standard or URL-safe alphabet, optional padding, >=16 chars
BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/_\-])[A-Za-z0-9+/_\-]{16,}={0,2}")
HEX_PLAIN_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}){8,}\b")
HEX_ESCAPED_RE = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){8,}")
HEX_SPACED_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[ :,]){7,}[0-9A-Fa-f]{2}\b")

# letters separated by a space / dot / dash / underscore / asterisk / pipe
SPACED_RE = re.compile(r"(?:\b[A-Za-z](?:[ .\-_*|]{1,3})){3,}[A-Za-z]\b")
LEET_TOKEN_RE = re.compile(r"\b(?=[A-Za-z0-9@$!|]*[A-Za-z])(?=[A-Za-z0-9@$!|]*[0-9@$!|])[A-Za-z0-9@$!|]{3,}\b")
LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                          "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "|": "l"})

# small function-word list used only to decide whether a transformed view
# (ROT13 / reversed) reads more like English than the original
_COMMON = set("""the be to of and a in that have i it for not on with he as you do at this
but his by from they we say her she or an will my one all would there their what so up
out if about who get which go me when make can like time no just him know take people
into year your good some could them see other than then now look only come its over
think also back after use two how our work first well way even new want because any
these give day most us is are was were been has had did does said please must should
ignore previous instructions instruction prompt system rules now act answer tell
""".split())

MAX_DECODE_DEPTH = 3


def _fold_homoglyphs(text: str, table: dict) -> str:
    return "".join(table.get(ch, ch) for ch in text)


def _strip_html(text: str) -> str:
    if "<" not in text and "&" not in text:
        return text
    try:
        soup = BeautifulSoup(text, "html.parser")
        for comment in soup.find_all(string=lambda s: s.strip().startswith("<!--")):
            comment.extract()
        return soup.get_text(separator=" ")
    except Exception:
        return text


def _looks_like_text(s: str) -> bool:
    """Accept a decoded payload only if it reads like natural-language text.
    Whitespace (including newlines/tabs) counts as printable, unlike
    str.isprintable(), which caused the legacy base64 failure."""
    if len(s) < 6:
        return False
    printable = sum(1 for c in s if c.isprintable() or c in "\n\r\t")
    if printable / len(s) < 0.95:
        return False
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return False
    letters = sum(1 for c in non_space if c.isalpha())
    if letters / len(non_space) < 0.6:
        return False
    return any(c.isspace() for c in s) or len(s) >= 12


_WORD_RE = re.compile(r"\b[a-z]{2,}\b")


def _englishness(s: str) -> float:
    toks = _WORD_RE.findall(s.lower())
    if len(toks) < 3:
        return 0.0
    known = sum(1 for t in toks if t in _COMMON)
    return known / len(toks) if known >= 3 else 0.0


def _known_words(s: str) -> int:
    return sum(1 for t in _WORD_RE.findall(s.lower()) if t in _COMMON)


def _try_b64(candidate: str):
    core = candidate.rstrip("=")
    # an ordinary word / identifier is not a payload: require mixed
    # character classes typical of base64 output
    has_lower = any(c.islower() for c in core)
    has_upper = any(c.isupper() for c in core)
    has_digit_or_sym = any(c.isdigit() or c in "+/-_" for c in core)
    if not (has_lower and has_upper) and not (has_digit_or_sym and (has_lower or has_upper)):
        return None
    if len(core) % 4 == 1:
        core = core[:-1]
    padded = core + "=" * (-len(core) % 4)
    alt = padded.replace("-", "+").replace("_", "/")
    try:
        raw = base64.b64decode(alt, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return s if _looks_like_text(s) else None


def _try_hex(candidate: str):
    digits = re.sub(r"\\x|[ :,]", "", candidate)
    if len(digits) % 2:
        return None
    try:
        s = bytes.fromhex(digits).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    return s if _looks_like_text(s) else None


def _decode_layer(text: str):
    out = []
    for m in BASE64_RE.finditer(text):
        d = _try_b64(m.group(0))
        if d:
            out.append(d)
    for rx in (HEX_ESCAPED_RE, HEX_SPACED_RE, HEX_PLAIN_RE):
        for m in rx.finditer(text):
            d = _try_hex(m.group(0))
            if d:
                out.append(d)
    return out


def _decode_recursive(text: str):
    found, frontier = [], [text]
    for _ in range(MAX_DECODE_DEPTH):
        nxt = []
        for t in frontier:
            for d in _decode_layer(t):
                if d not in found:
                    found.append(d)
                    nxt.append(d)
        if not nxt:
            break
        frontier = nxt
    return found


def _collapse_spaced(match: str) -> str:
    if re.search(r"[.\-_*|]", match):
        words = match.split()
        return " ".join(re.sub(r"[.\-_*|]", "", w) for w in words)
    words = re.split(r"\s{2,}", match)
    return " ".join(w.replace(" ", "") for w in words)


def _deobfuscation_views(text: str):
    views = []
    # spaced / dotted letters
    spaced = [_collapse_spaced(m.group(0)) for m in SPACED_RE.finditer(text)]
    spaced = [s for s in spaced if len(s.replace(" ", "")) >= 4]
    if spaced:
        views.append(" ".join(spaced))
    # leetspeak: only rewrite tokens that mix letters with digits/symbols,
    # and only emit the sentences that actually changed (keeps benign text
    # containing e.g. "GPT4" from being duplicated wholesale)
    if LEET_TOKEN_RE.search(text):
        changed = []
        for sent in re.split(r"(?<=[.!?\n])\s+", text):
            leet = LEET_TOKEN_RE.sub(lambda m: m.group(0).translate(LEET_MAP).lower(), sent)
            # keep only if de-leeting produced more recognisable words
            if leet != sent and _known_words(leet) > _known_words(sent):
                changed.append(leet)
        if changed:
            views.append(" ".join(changed))
    # ROT13 and reversal: keep only if clearly more English than the input
    base = _englishness(text)
    for transformed in (codecs.decode(text, "rot13"), text[::-1]):
        e = _englishness(transformed)
        if e >= 0.3 and e > base + 0.15:
            views.append(transformed)
    return views


def _sanitize_legacy(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _LEGACY_ZERO_WIDTH.sub("", text)
    text = _fold_homoglyphs(text, _LEGACY_HOMOGLYPHS)
    text = _strip_html(text)
    chunks = []
    for match in _LEGACY_BASE64_RE.finditer(text):
        try:
            decoded = base64.b64decode(match.group(0), validate=True).decode("utf-8", errors="ignore")
            if decoded.isprintable() and len(decoded) > 8:
                chunks.append(decoded)
        except (binascii.Error, ValueError):
            continue
    if chunks:
        text = text + " [DECODED] " + " ".join(chunks)
    return re.sub(r"\s+", " ", text).strip()


def _sanitize_full(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    text = _fold_homoglyphs(text, HOMOGLYPHS)
    text = _strip_html(text)
    decoded = _decode_recursive(text)
    views = []
    for v in [text] + decoded:
        for view in _deobfuscation_views(v):
            if view not in views and view not in decoded:
                views.append(view)

    def _clean(e):
        return _fold_homoglyphs(ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", e)), HOMOGLYPHS)

    # "[DECODED]" marks a decoded encoding (feeds the has_encoded_blob
    # feature); "[DEOBF]" marks a de-obfuscation view (leet/spaced/rot13/rev)
    if decoded:
        text = text + " [DECODED] " + " ".join(_clean(e) for e in decoded)
    if views:
        text = text + " [DEOBF] " + " ".join(_clean(e) for e in views)
    return re.sub(r"\s+", " ", text).strip()


def sanitize(text: str, mode: str = "full") -> str:
    if not isinstance(text, str):
        text = str(text)
    if mode == "legacy":
        return _sanitize_legacy(text)
    if mode == "none":
        return re.sub(r"\s+", " ", text).strip()
    return _sanitize_full(text)
