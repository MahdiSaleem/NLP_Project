"""PAW (Pieces of Arabic Words) tokenizer for Part C v3.

Reads `Dataset/LegalAmounts_tokenized/*.txt` whose lines look like:

    Lac00000.tif \\u202b['ثمانيه', 'و', 'اربع', ...]\\u202c

Builds a train-only PAW vocab. ID 0 is the CTC blank; IDs 1..N-1 are PAW chunks
(plus a final `<unk>` for OOV PAWs in val/test). Encoding/decoding helpers are
provided. Save/load JSON.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

UNK = "<unk>"
RTL_LTR = ["‫", "‬", "‎", "‏", "‪", "‭", "‮"]


def strip_bidi(text: str) -> str:
    for c in RTL_LTR:
        text = text.replace(c, "")
    return text.strip()


def parse_paw_file(filepath: Path) -> dict[str, list[str]]:
    """Return {image_stem: [paw_chunks]} for one tokenized file."""
    out: dict[str, list[str]] = {}
    for line in filepath.read_text(encoding="utf-8").splitlines():
        line = strip_bidi(line.strip())
        if not line:
            continue
        # Split filename from list. Filename is the first whitespace-delimited token.
        m = re.match(r"^(\S+)\s+(\[.*\])\s*$", line)
        if not m:
            continue
        raw_id, list_str = m.group(1), m.group(2)
        if raw_id.startswith("L"):
            raw_id = raw_id[1:]
        image_stem = Path(raw_id).stem
        try:
            paws = ast.literal_eval(list_str)
        except Exception:
            continue
        if not isinstance(paws, list) or not all(isinstance(p, str) for p in paws):
            continue
        out[image_stem] = [strip_bidi(p) for p in paws if strip_bidi(p)]
    return out


def load_paw_dir(tok_dir: Path) -> dict[str, list[str]]:
    """Read all tokenized files into one {image_stem: [paws]} dict."""
    out: dict[str, list[str]] = {}
    for fp in sorted(Path(tok_dir).glob("*.txt")):
        out.update(parse_paw_file(fp))
    return out


class PAWTokenizer:
    def __init__(self, paws: list[str]) -> None:
        # ID 0 reserved for CTC blank; IDs 1..N for PAW chunks; last ID for <unk>.
        # Stable order: blank, then sorted train PAWs, then <unk>.
        self.blank_id = 0
        ordered = ["<blank>"] + sorted(paws) + [UNK]
        if len(set(ordered)) != len(ordered):
            raise ValueError("Duplicate tokens (UNK collision with a real PAW?)")
        self.id_to_paw = {i: t for i, t in enumerate(ordered)}
        self.paw_to_id = {t: i for i, t in enumerate(ordered)}
        self.unk_id = self.paw_to_id[UNK]
        self.vocab_size = len(ordered)
        self.train_paws = list(sorted(paws))

    def encode(self, paws: list[str]) -> list[int]:
        return [self.paw_to_id.get(p, self.unk_id) for p in paws]

    def decode(self, ids: list[int], drop_blank: bool = True, drop_unk: bool = False) -> list[str]:
        out: list[str] = []
        for i in ids:
            if drop_blank and i == self.blank_id:
                continue
            tok = self.id_to_paw.get(int(i), UNK)
            if drop_unk and tok == UNK:
                continue
            if tok == "<blank>":
                continue
            out.append(tok)
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"train_paws": self.train_paws}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "PAWTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(paws=data["train_paws"])


def join_paws(paws: list[str]) -> str:
    return " ".join(paws)
