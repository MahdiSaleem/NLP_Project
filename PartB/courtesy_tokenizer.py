"""Tokenizer for courtesy-amount labels in Dataset/CourtesyAmounts/*.txt.

Each label file contains lines of `Cac<id>.tif\\t<python_list_literal>`, e.g.
    Cac00000.tif    [10, 4, 8, 6, 2, 6, 10]
where the wrapping `10`s are start/end markers (NOT digits). Inner tokens are
ints 0-9 or single-char strings '.' (decimal) or '/' (fraction-bar).

We coerce each inner token to a 1-char string and join — a digit sequence is
naturally a string (e.g. "48626", "6242.60"). This lets us reuse Part C's
char-level CRNN+CTC pipeline directly with a 12-symbol vocab.

Per the project PDF, the final output should be "digits without any special
characters" — `to_digits_only(s)` strips `.` and `/` for metric/output.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

START_END_MARKER = 10
SPECIAL_DIGIT_CHARS = (".", "/")


def parse_courtesy_file(path: Path) -> dict[str, str]:
    """Parse one courtesy label file → {image_stem: digit_string}.

    image_stem: 'ac00000' (no leading C, no .tif).
    digit_string: e.g. '48626' or '6242.60'. Wrapping 10s are stripped.
    """
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            raw_id, lit = parts
            if raw_id.startswith("C"):
                raw_id = raw_id[1:]
            stem = Path(raw_id).stem
            try:
                tokens = ast.literal_eval(lit.strip())
            except (ValueError, SyntaxError):
                continue
            if not isinstance(tokens, (list, tuple)):
                continue
            # Strip leading/trailing 10 markers.
            toks = list(tokens)
            while toks and toks[0] == START_END_MARKER:
                toks.pop(0)
            while toks and toks[-1] == START_END_MARKER:
                toks.pop()
            if not toks:
                continue
            chars: list[str] = []
            for t in toks:
                if isinstance(t, int):
                    if 0 <= t <= 9:
                        chars.append(str(t))
                    else:
                        # Stray 10 in middle is unexpected; drop it (rare).
                        continue
                elif isinstance(t, str) and len(t) == 1:
                    chars.append(t)
            if chars:
                out[stem] = "".join(chars)
    return out


def load_all_courtesy(label_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for fp in sorted(label_dir.glob("*.txt")):
        out.update(parse_courtesy_file(fp))
    return out


def build_vocab(train_strings: list[str]) -> dict:
    """Build vocab from train labels only (honest baseline). blank=0.
    Returns {char_to_idx, idx_to_char (str-keys), size, blank}.
    """
    counts: Counter[str] = Counter()
    for s in train_strings:
        counts.update(s)
    chars = sorted(counts.keys())
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 reserved for blank
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    return {
        "blank": 0,
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "size": len(chars) + 1,
        "char_counts": dict(counts),
    }


def save_vocab(vocab: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> tuple[dict[str, int], dict[int, str], int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    char_to_idx = data["char_to_idx"]
    idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
    return char_to_idx, idx_to_char, data["size"]


def to_digits_only(s: str) -> str:
    """Strip '.' and '/' (and any other non-digit). Per PDF: 'sequence of digits
    without any special characters'.
    """
    return "".join(c for c in s if c.isdigit())
