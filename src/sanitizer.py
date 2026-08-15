"""
Stage 1: Sanitization.

Normalizes raw text before analysis so that later stages see a canonical
form rather than an obfuscated one:
  - Unicode NFKC normalization + zero-width character stripping
  - Homoglyph folding for a small set of commonly abused characters
  - HTML/markdown tag and comment stripping (relevant for indirect input
    pulled from web pages / documents)
  - Detection and decoding of long base64-looking substrings so hidden
    payloads are not missed by keyword/pattern analysis
"""
import re
import base64
import binascii
import unicodedata
from bs4 import BeautifulSoup

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")

# a small homoglyph table covering common lookalike substitutions
HOMOGLYPHS = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0443": "y", "\u0456": "i", "\u0501": "d", "\u04bb": "h", "\u0455": "s",
}

BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


def _fold_homoglyphs(text: str) -> str:
    return "".join(HOMOGLYPHS.get(ch, ch) for ch in text)


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


def _decode_base64_blobs(text: str) -> str:
    """Append decoded content of any long base64-looking substrings so the
    keyword stage can inspect what was hidden, without discarding the
    original (still-suspicious) encoded text."""
    decoded_chunks = []
    for match in BASE64_RE.finditer(text):
        candidate = match.group(0)
        try:
            decoded = base64.b64decode(candidate, validate=True)
            decoded_text = decoded.decode("utf-8", errors="ignore")
            if decoded_text.isprintable() and len(decoded_text) > 8:
                decoded_chunks.append(decoded_text)
        except (binascii.Error, ValueError):
            continue
    if decoded_chunks:
        return text + " [DECODED] " + " ".join(decoded_chunks)
    return text


def sanitize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    text = _fold_homoglyphs(text)
    text = _strip_html(text)
    text = _decode_base64_blobs(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
