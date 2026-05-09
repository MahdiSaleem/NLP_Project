"""Rule-based Arabic-text → integer parser for Part D.

Inputs come from two sources:
  1. Whitespace-tokenized raw legal text (`parse_legal(text)`).
  2. PAW lists from TrOCR-PAW (`parse_paw_list(paws)`) — joined into a
     contiguous string then re-segmented via greedy longest-match.

All vocabulary keys are stored in normalized form (see normalize.py).

Returns ParseResult with `value` (int|None), `valid` (bool), and `leftover`
(unmapped tokens). The verifier uses `valid + value` to emit MATCH/MISMATCH/
UNPARSED labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from PartD.normalize import normalize_token

# ---------------------------------------------------------------------------
# Number-word tables. All keys are POST-normalization.
# ---------------------------------------------------------------------------

# Units 0-10 (cardinal feminine, the form used for currency in this dataset).
_UNITS_RAW = {
    0: ["صفر"],
    1: ["واحد", "واحده", "احد", "احدي", "احدا", "احدى", "إحدى"],
    2: ["اثنان", "اثنين", "اثنا", "اثنتان", "اثنتين", "اثني", "اثنى", "اثن"],
    3: ["ثلاثه", "ثلاث"],
    4: ["اربعه", "اربع"],
    5: ["خمسه", "خمس"],
    6: ["سته", "ست", "ستت"],
    7: ["سبعه", "سبع"],
    8: ["ثمانيه", "ثماني", "ثمان"],
    9: ["تسعه", "تسع"],
    10: ["عشره", "عشر"],
}

# Tens 20-90.
_TENS_RAW = {
    20: ["عشرين", "عشرون"],
    30: ["ثلاثين", "ثلاثون"],
    40: ["اربعين", "اربعون", "ربعون", "ربعين"],  # last two are informal
    50: ["خمسين", "خمسون"],
    60: ["ستين", "ستون"],
    70: ["سبعين", "سبعون"],
    80: ["ثمانين", "ثمانون"],
    90: ["تسعين", "تسعون"],
}

# Hundreds. Includes glued, split-with-elongation, and dialectal forms.
_HUNDREDS_RAW = {
    100: ["مائه", "مئه", "ميه"],
    200: ["مائتان", "مائتين", "مئتان", "مئتين", "ميتان", "ميتين", "ماتين"],
    300: ["ثلاثمائه", "ثلاثمئه", "ثلثمائه", "ثلثمئه", "ثلثميه", "ثلاثميه", "ثلاثايه"],
    400: ["اربعمائه", "اربعمئه", "اربعميه", "اربعاميه", "اربعايه"],
    500: ["خمسمائه", "خمسمئه", "خمسميه", "خمساميه", "خمسامئه", "خمسايه"],
    600: ["ستمائه", "ستمئه", "ستميه", "ستاميه", "ستايه"],
    700: ["سبعمائه", "سبعمئه", "سبعميه", "سبعاميه", "سبعايه"],
    800: ["ثمانمائه", "ثمانيمائه", "ثمانمئه", "ثمانميه", "ثمانيميه", "ثمانايه"],
    900: ["تسعمائه", "تسعمئه", "تسعميه", "تسعاميه", "تسعايه"],
}

# Thousand literal singletons.
_THOUSAND_LIT_RAW = {
    1000: ["الف", "الفا", "الفاً", "ألفاً", "ألفا", "االف"],  # last is typo
    2000: ["الفان", "الفين", "ألفان", "ألفين"],
}

# Thousand multiplier word — combines with a preceding number value.
_THOUSAND_MULT_RAW = ["الاف", "آلاف", "ألاف"]  # all normalize to الاف

# Million literal singletons.
_MILLION_LIT_RAW = {
    1_000_000: ["مليون", "مليونا", "مليوناً"],
    2_000_000: ["مليونان", "مليونين"],
}

# Million multiplier (rare).
_MILLION_MULT_RAW = ["ملايين", "ملاين"]

# Frame / non-numeric words to strip (currency markers, "only", etc.).
_FRAME_RAW = [
    "ريال", "ريالا", "ريالات", "ريالاً", "ريالان",
    "فقط", "لا", "غير", "لاغير",
    "سعودي", "سعوديه",
    "هلله", "هلل",
    "وقدره", "قدره", "قدرها", "قدر",
]

# Cents/fraction markers (e.g. "٦٣/١٠٠"). Pattern, not exact-string lookup.
_CENTS_FRAC_RE = re.compile(r"[٠-٩0-9]+\s*[/⁄]\s*100")
_CENTS_INT_RE = re.compile(r"^[٠-٩0-9]{1,2}$")

# Map Eastern-Arabic digits to ASCII.
_EAST_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _build_lookup(raw: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for v, forms in raw.items():
        for f in forms:
            out[normalize_token(f)] = v
    return out


def _build_set(raw: list[str]) -> set[str]:
    return {normalize_token(f) for f in raw if f}


UNITS = _build_lookup(_UNITS_RAW)
TENS = _build_lookup(_TENS_RAW)
HUNDREDS = _build_lookup(_HUNDREDS_RAW)
THOUSAND_LIT = _build_lookup(_THOUSAND_LIT_RAW)
THOUSAND_MULT = _build_set(_THOUSAND_MULT_RAW)
MILLION_LIT = _build_lookup(_MILLION_LIT_RAW)
MILLION_MULT = _build_set(_MILLION_MULT_RAW)
FRAME = _build_set(_FRAME_RAW)

# Riyal markers — used to split integer (left) from cents/halalas (right).
_RIYAL_RAW = ["ريال", "ريالا", "ريالاً", "ريالان", "ريالات", "ريالا"]
RIYAL = _build_set(_RIYAL_RAW)

# Halala markers — strip but track that we're in cents territory.
_HALALA_RAW = ["هلله", "هللة", "هلل", "هللات"]
HALALA = _build_set(_HALALA_RAW)

# Combined number-word vocab — for greedy re-segmentation.
NUMBER_VOCAB = set()
NUMBER_VOCAB.update(UNITS)
NUMBER_VOCAB.update(TENS)
NUMBER_VOCAB.update(HUNDREDS)
NUMBER_VOCAB.update(THOUSAND_LIT)
NUMBER_VOCAB.update(THOUSAND_MULT)
NUMBER_VOCAB.update(MILLION_LIT)
NUMBER_VOCAB.update(MILLION_MULT)
NUMBER_VOCAB.add(normalize_token("و"))


@dataclass
class ParseResult:
    value: int | None
    cents: int | None = None
    valid: bool = False
    leftover: list[str] = field(default_factory=list)
    normalized: str = ""


# ---------------------------------------------------------------------------
# Parser core.
# ---------------------------------------------------------------------------


def _strip_leading_waw(t: str) -> tuple[str, bool]:
    """Strip a single leading و (and). Returns (token, had_waw)."""
    if t.startswith("و") and len(t) > 1 and t[1:] != "و":
        return t[1:], True
    return t, False


def _classify(t: str):
    """Return (kind, value) — kind in {'unit','ten','hundred','thousand_lit',
    'thousand_mult','million_lit','million_mult','waw','frame', None}."""
    if t in FRAME:
        return ("frame", 0)
    if t == "و":
        return ("waw", 0)
    if t in UNITS:
        return ("unit", UNITS[t])
    if t in TENS:
        return ("ten", TENS[t])
    if t in HUNDREDS:
        return ("hundred", HUNDREDS[t])
    if t in THOUSAND_LIT:
        return ("thousand_lit", THOUSAND_LIT[t])
    if t in THOUSAND_MULT:
        return ("thousand_mult", 0)
    if t in MILLION_LIT:
        return ("million_lit", MILLION_LIT[t])
    if t in MILLION_MULT:
        return ("million_mult", 0)
    return (None, 0)


def _preclean(tokens: list[str]) -> tuple[list[str], list[str], int | None, bool]:
    """First pass: split on the FIRST `ريال` boundary, drop frame words,
    split leading و, collect cents fractions. Tracks whether ANY halala
    word appears on the right side — that disambiguates whether right-side
    numbers are cents (halala present) or part of the integer (no halala).

    Returns (left_tokens, right_tokens, cents_explicit, has_halala).
    """
    left: list[str] = []
    right: list[str] = []
    cents: int | None = None
    seen_riyal = False
    has_halala = False

    def _push(side: list[str], t: str) -> None:
        if not t:
            return
        if t in FRAME:
            return
        side.append(t)

    for raw in tokens:
        t = normalize_token(raw)
        if not t:
            continue

        m = _CENTS_FRAC_RE.search(t)
        if m:
            digits = re.findall(r"[٠-٩0-9]+", m.group(0))[0].translate(_EAST_DIG)
            try:
                cents = int(digits)
            except ValueError:
                pass
            continue
        if _CENTS_INT_RE.match(t):
            try:
                v = int(t.translate(_EAST_DIG))
                if 0 <= v <= 99:
                    cents = v
                    continue
            except ValueError:
                pass

        if t in RIYAL:
            seen_riyal = True
            continue
        if t in HALALA:
            has_halala = True
            continue
        stripped, had_waw = _strip_leading_waw(t)
        if had_waw and stripped in RIYAL:
            seen_riyal = True
            continue
        if had_waw and stripped in HALALA:
            has_halala = True
            continue
        if had_waw and (stripped in NUMBER_VOCAB or stripped in FRAME):
            target = right if seen_riyal else left
            _push(target, "و")
            _push(target, stripped)
            continue

        target = right if seen_riyal else left
        _push(target, t)

    return left, right, cents, has_halala


def _parse_chunk(toks: list[str]) -> tuple[int, list[str], bool]:
    """Run the scale-based number-word grammar on a list of normalized tokens.
    Returns (value, leftover, saw_any_number).

    Algorithm:
      `total` accumulates scaled values. `group` is the currently-being-built
      sub-1000 value. Thousand-scale tokens commit `group * 1000` (default
      group=1 for a bare singular) to total and reset group; the same idea at
      million scale. `الفان` = 2000 / `مليونان` = 2,000,000 are literal
      additions. Peephole: a unit u in {3..9} immediately followed by a bare
      `مايه` (100) is rewritten as u*100 (covers split forms like "ثلاث مائه").
    """
    total = 0
    group = 0
    saw = False
    leftover: list[str] = []
    last_unit: int | None = None  # value last emitted as 'unit'

    for t in toks:
        kind, v = _classify(t)
        if kind in ("frame", "waw"):
            last_unit = None
            continue
        if kind == "unit":
            group += v
            saw = True
            last_unit = v
            continue
        if kind == "ten":
            group += v
            saw = True
            last_unit = None
            continue
        if kind == "hundred":
            # Split-form peephole: `<unit 3-9> مايه` → unit*100.
            if v == 100 and last_unit is not None and 3 <= last_unit <= 9:
                group -= last_unit
                group += last_unit * 100
            else:
                group += v
            saw = True
            last_unit = None
            continue
        if kind == "thousand_lit":
            if v == 1000:
                total += (group if group > 0 else 1) * 1000
                group = 0
            else:  # 2000
                total += v
                group = 0
            saw = True
            last_unit = None
            continue
        if kind == "thousand_mult":
            total += (group if group > 0 else 1) * 1000
            group = 0
            saw = True
            last_unit = None
            continue
        if kind == "million_lit":
            if v == 1_000_000:
                # The whole "everything below" block multiplies the million.
                lower = total + group
                if lower == 0:
                    lower = 1
                total = lower * 1_000_000
                group = 0
            else:  # 2 million
                total += v
                group = 0
            saw = True
            last_unit = None
            continue
        if kind == "million_mult":
            lower = total + group
            if lower == 0:
                lower = 1
            total = lower * 1_000_000
            group = 0
            saw = True
            last_unit = None
            continue
        leftover.append(t)
        last_unit = None

    return total + group, leftover, saw


def parse_tokens(tokens: list[str]) -> ParseResult:
    """Top-level: split on first `ريال`. If halala/cents context is detected
    on the right side, parse it as cents; otherwise fold the right side into
    the integer (handles checks that simply repeat ريال mid-amount).
    """
    left, right, cents_explicit, has_halala = _preclean(tokens)
    integer_value, leftover_left, saw_left = _parse_chunk(left)
    cents_value: int | None = cents_explicit
    leftover_right: list[str] = []

    if right:
        c, leftover_right, saw_right = _parse_chunk(right)
        if has_halala or cents_explicit is not None:
            # Cents context — keep cents, do NOT fold into integer.
            if cents_value is None and saw_right and 0 <= c <= 99:
                cents_value = c
            elif cents_value is None and saw_right and c > 99:
                # Surfaces as a diagnostic — likely bad text; leave cents None.
                leftover_right.append(f"<right={c}>")
        else:
            # No halala marker → right is more integer (e.g. "ألف ريال واحد").
            if saw_right:
                integer_value += c

    valid = saw_left and len(leftover_left) <= 2
    return ParseResult(
        value=(integer_value if valid else None),
        cents=cents_value,
        valid=valid,
        leftover=leftover_left + leftover_right,
        normalized=" ".join(left) + (" | " + " ".join(right) if right else ""),
    )


def parse_legal(text: str) -> ParseResult:
    """Parse free Arabic text → integer."""
    return parse_tokens(text.split())


# ---------------------------------------------------------------------------
# PAW list -> string -> word re-segmentation.
# ---------------------------------------------------------------------------


def _greedy_resegment(joined: str, vocab: set[str], max_len: int = 12) -> list[str]:
    """Greedy longest-match left-to-right re-segmentation.

    Scan `joined` (already normalized, no whitespace). At each position,
    try the longest substring (up to `max_len`) that is in `vocab`. If
    nothing matches at a position, advance one char and emit the leftover
    later (collected as a leftover token).
    """
    if not joined:
        return []
    out: list[str] = []
    i = 0
    n = len(joined)
    leftover_buf = ""
    while i < n:
        matched = None
        upper = min(max_len, n - i)
        for L in range(upper, 0, -1):
            cand = joined[i:i + L]
            if cand in vocab:
                matched = cand
                break
        if matched is None:
            leftover_buf += joined[i]
            i += 1
            continue
        if leftover_buf:
            out.append(leftover_buf)
            leftover_buf = ""
        out.append(matched)
        i += len(matched)
    if leftover_buf:
        out.append(leftover_buf)
    return out


def parse_paw_list(paws: list[str]) -> ParseResult:
    """Join PAWs (no whitespace), normalize, then greedy re-segment."""
    joined = "".join(normalize_token(p) for p in paws if p)
    # Build the segmentation vocab — number words + frame words (so frame
    # words get peeled off too) + و.
    vocab = set(NUMBER_VOCAB)
    vocab.update(FRAME)
    tokens = _greedy_resegment(joined, vocab)
    return parse_tokens(tokens)
