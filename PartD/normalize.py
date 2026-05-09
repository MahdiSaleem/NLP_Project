"""Arabic text normalization for Part D parser.

The legal-amount labels in this dataset use many spelling variants of the
same number-word (e.g. ثلاثة vs ثلاثه, أربعة vs اربعه). This module
canonicalizes them so a single lookup table covers all variants.
"""
from __future__ import annotations

import re
import unicodedata

# Bidi marks that appear in tokenized labels.
_BIDI_CHARS = "‪‫‬‭‮‎‏؜"
# Tatweel (kashida) — pure visual elongation.
_TATWEEL = "ـ"
# Diacritics: fathatan, dammatan, kasratan, fatha, damma, kasra, shadda, sukun,
# superscript alef.
_DIACRITICS = "ًٌٍَُِّْٰ"

# Build a unified strip pattern.
_STRIP_RE = re.compile(f"[{_BIDI_CHARS}{_TATWEEL}{_DIACRITICS}]")

# Letter-form unification.
_LETTER_MAP = str.maketrans({
    "ة": "ه",   # ta marbuta -> ha
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
})


def normalize_token(t: str) -> str:
    """Canonicalize one token: drop bidi/tatweel/diacritics, unify letter forms."""
    if not t:
        return ""
    # NFC just to be safe (combine pre-decomposed forms).
    t = unicodedata.normalize("NFC", t)
    t = _STRIP_RE.sub("", t)
    t = t.translate(_LETTER_MAP)
    return t.strip()


def normalize_text(s: str) -> str:
    """Tokenize on whitespace, normalize each token, rejoin with single spaces."""
    return " ".join(normalize_token(p) for p in s.split() if normalize_token(p))
