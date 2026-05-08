"""Minimal char-level tokenizer for the TrOCR decoder fallback path.

Builds a vocab of {<pad>, <bos>, <eos>, <unk>} + the train-only chars from
`PartC/vocab.json` (the same 60-char set the CRNN uses). Supports the subset of
the HF tokenizer API that `trocr_dataset.py` needs:

  - __call__(text, padding="max_length", max_length=N, truncation=True, return_tensors="pt")
    returns an object with `.input_ids` of shape (1, max_length).
  - batch_decode(ids, skip_special_tokens=True) -> list[str]
  - pad_token_id, bos_token_id, eos_token_id, unk_token_id, vocab_size
  - save(out_dir) / load(out_dir)

This is NOT a HuggingFace `PreTrainedTokenizer`; it just quacks like one in the
ways our pipeline cares about. Avoids the AraBERT/sentencepiece glue that has
known fine-tuning failure modes (HF issue #19329).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


@dataclass
class _EncodeOutput:
    input_ids: torch.Tensor


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        # Specials first so their IDs are stable: pad=0, bos=1, eos=2, unk=3.
        all_tokens = list(SPECIAL_TOKENS) + list(chars)
        # Validate uniqueness (chars must not overlap with specials).
        if len(set(all_tokens)) != len(all_tokens):
            raise ValueError("Char vocab contains duplicates or overlaps with special tokens")
        self.id_to_token = {i: t for i, t in enumerate(all_tokens)}
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.chars = list(chars)

        self.pad_token_id = self.token_to_id[PAD]
        self.bos_token_id = self.token_to_id[BOS]
        self.eos_token_id = self.token_to_id[EOS]
        self.unk_token_id = self.token_to_id[UNK]
        self.vocab_size = len(all_tokens)

        self._special_ids = {
            self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id,
        }

    def encode_text(self, text: str) -> list[int]:
        out = [self.bos_token_id]
        for c in text:
            out.append(self.token_to_id.get(c, self.unk_token_id))
        out.append(self.eos_token_id)
        return out

    def __call__(
        self,
        text: str,
        padding: str = "max_length",
        max_length: int = 128,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> _EncodeOutput:
        ids = self.encode_text(text)
        if truncation and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.eos_token_id]
        if padding == "max_length":
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
        if return_tensors == "pt":
            return _EncodeOutput(input_ids=torch.tensor([ids], dtype=torch.long))
        return _EncodeOutput(input_ids=ids)  # type: ignore[arg-type]

    def batch_decode(self, ids_batch, skip_special_tokens: bool = True) -> list[str]:
        if hasattr(ids_batch, "tolist"):
            ids_batch = ids_batch.tolist()
        results: list[str] = []
        for ids in ids_batch:
            chars: list[str] = []
            for tid in ids:
                tid = int(tid)
                if tid == self.eos_token_id:
                    break
                if skip_special_tokens and tid in self._special_ids:
                    continue
                chars.append(self.id_to_token.get(tid, ""))
            results.append("".join(chars))
        return results

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"chars": self.chars}
        (out_dir / "char_tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, in_dir: Path) -> "CharTokenizer":
        payload = json.loads((Path(in_dir) / "char_tokenizer.json").read_text(encoding="utf-8"))
        return cls(chars=payload["chars"])

    @classmethod
    def from_partc_vocab(cls, vocab_json_path: Path) -> "CharTokenizer":
        """Build from PartC/vocab.json (CRNN char_to_idx, blank=0)."""
        data = json.loads(Path(vocab_json_path).read_text(encoding="utf-8"))
        chars = list(data["char_to_idx"].keys())
        return cls(chars=chars)


class TrOCRCharProcessor:
    """Minimal stand-in for `TrOCRProcessor` that keeps the original image_processor
    but uses our `CharTokenizer` as `.tokenizer`. Quacks enough to be passed into
    `trocr_dataset.py`'s `TrOCRLegalLineDataset`.
    """

    def __init__(self, image_processor, tokenizer: CharTokenizer) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images=None, return_tensors="pt"):
        return self.image_processor(images=images, return_tensors=return_tensors)

    def batch_decode(self, ids_batch, skip_special_tokens: bool = True) -> list[str]:
        return self.tokenizer.batch_decode(ids_batch, skip_special_tokens=skip_special_tokens)

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.image_processor.save_pretrained(out_dir)
        self.tokenizer.save(out_dir)
