"""HF-compatible adapter for PAW tokenization in the TrOCR decoder.

Wraps PartC.paw_tokenizer.PAWTokenizer with the subset of the HF tokenizer API
that trocr_dataset.py and train_trocr.py expect. Treats each PAW as one decoder
token. Vocab layout: {<pad>=0, <bos>=1, <eos>=2, <unk>=3} ++ sorted train PAWs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from PartC.paw_tokenizer import PAWTokenizer

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK_DEC = "<unk_dec>"  # decoder UNK; distinct from PAWTokenizer's content UNK
SPECIALS = [PAD, BOS, EOS, UNK_DEC]


@dataclass
class _EncodeOutput:
    input_ids: torch.Tensor


class PAWDecoderTokenizer:
    """Treats each PAW chunk as one decoder token.

    Note: this is independent of PAWTokenizer (which is for CTC and uses ID 0 as
    blank). The decoder version uses ID 0 as <pad>, and adds BOS/EOS specials.
    """

    def __init__(self, train_paws: list[str]) -> None:
        self.train_paws = sorted(set(train_paws))
        all_tokens = SPECIALS + self.train_paws
        if len(set(all_tokens)) != len(all_tokens):
            raise ValueError("PAW vocab overlaps a special token")
        self.id_to_token = {i: t for i, t in enumerate(all_tokens)}
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.pad_token_id = self.token_to_id[PAD]
        self.bos_token_id = self.token_to_id[BOS]
        self.eos_token_id = self.token_to_id[EOS]
        self.unk_token_id = self.token_to_id[UNK_DEC]
        self.vocab_size = len(all_tokens)
        self._special_ids = {self.pad_token_id, self.bos_token_id,
                             self.eos_token_id, self.unk_token_id}

    def encode_paws(self, paws: list[str]) -> list[int]:
        out = [self.bos_token_id]
        for p in paws:
            out.append(self.token_to_id.get(p, self.unk_token_id))
        out.append(self.eos_token_id)
        return out

    def __call__(
        self,
        paws: list[str],
        padding: str = "max_length",
        max_length: int = 64,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> _EncodeOutput:
        ids = self.encode_paws(paws)
        if truncation and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.eos_token_id]
        if padding == "max_length":
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
        if return_tensors == "pt":
            return _EncodeOutput(input_ids=torch.tensor([ids], dtype=torch.long))
        return _EncodeOutput(input_ids=ids)  # type: ignore[arg-type]

    def batch_decode(self, ids_batch, skip_special_tokens: bool = True) -> list[list[str]]:
        if hasattr(ids_batch, "tolist"):
            ids_batch = ids_batch.tolist()
        out: list[list[str]] = []
        for ids in ids_batch:
            paws: list[str] = []
            for tid in ids:
                tid = int(tid)
                if tid == self.eos_token_id:
                    break
                if skip_special_tokens and tid in self._special_ids:
                    continue
                paws.append(self.id_to_token.get(tid, UNK_DEC))
            out.append(paws)
        return out

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paw_dec_tokenizer.json").write_text(
            json.dumps({"train_paws": self.train_paws}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, in_dir: Path) -> "PAWDecoderTokenizer":
        path = Path(in_dir) / "paw_dec_tokenizer.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(train_paws=data["train_paws"])

    @classmethod
    def from_paw_vocab(cls, paw_vocab_json: Path) -> "PAWDecoderTokenizer":
        ctc_tok = PAWTokenizer.load(paw_vocab_json)
        return cls(train_paws=ctc_tok.train_paws)


class TrOCRPAWProcessor:
    """Mirror of TrOCRCharProcessor but with PAW-token-level tokenizer."""

    def __init__(self, image_processor, tokenizer: PAWDecoderTokenizer) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images=None, return_tensors="pt"):
        return self.image_processor(images=images, return_tensors=return_tensors)

    def batch_decode(self, ids_batch, skip_special_tokens: bool = True) -> list[list[str]]:
        return self.tokenizer.batch_decode(ids_batch, skip_special_tokens=skip_special_tokens)

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.image_processor.save_pretrained(out_dir)
        self.tokenizer.save(out_dir)
