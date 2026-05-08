"""Inspect why the existing TrOCR fine-tune produces 100% val CER.

Loads `--model-dir` (a HF directory written by train_trocr.py), runs both greedy
and beam generation on a few val crops, prints raw token IDs + decoded strings,
and compares them to the GT char IDs. Flags common failure modes:

  - decoder_start_token_id mismatch between `model.config` and `model.generation_config`
  - generation collapsing to <pad>/<eos> immediately
  - beam search at small vocab + high `no_repeat_ngram_size` killing all hypotheses
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, VisionEncoderDecoderModel

from PartC.char_tokenizer import CharTokenizer, TrOCRCharProcessor
from PartC.decode import cer_wer
from PartC.trocr_dataset import TrOCRLegalLineDataset, collate_trocr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--crops-dir", type=Path, default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    crops_root = args.crops_dir or (data_dir / "crops" / "gt")

    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    char_tok = CharTokenizer.load(args.model_dir)
    processor = TrOCRCharProcessor(image_processor=image_processor, tokenizer=char_tok)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    print("=" * 60)
    print("CONFIG INSPECTION")
    print("=" * 60)
    print(f"vocab_size      : {processor.tokenizer.vocab_size}")
    print(f"pad_token_id    : {char_tok.pad_token_id}")
    print(f"bos_token_id    : {char_tok.bos_token_id}")
    print(f"eos_token_id    : {char_tok.eos_token_id}")
    print(f"unk_token_id    : {char_tok.unk_token_id}")
    print()
    print("model.config:")
    for k in ("pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id", "vocab_size"):
        print(f"  {k:30s} {getattr(model.config, k, '<missing>')}")
    if hasattr(model.config, "decoder"):
        print("model.config.decoder:")
        for k in ("pad_token_id", "bos_token_id", "eos_token_id", "vocab_size"):
            print(f"  {k:30s} {getattr(model.config.decoder, k, '<missing>')}")
    print("model.generation_config:")
    gc = model.generation_config
    for k in ("pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id",
              "max_length", "num_beams", "early_stopping", "no_repeat_ngram_size"):
        print(f"  {k:30s} {getattr(gc, k, '<missing>')}")
    print()

    ds = TrOCRLegalLineDataset(
        data_dir / "splits" / f"{args.split}.json",
        crops_root / args.split,
        processor,
        max_label_len=128,
    )
    ds.records = ds.records[: args.num_samples]
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_trocr)

    print("=" * 60)
    print("PER-SAMPLE GENERATION TRACE")
    print("=" * 60)
    for batch in loader:
        gt = batch["texts"][0]
        pixel_values = batch["pixel_values"].to(device)
        with torch.no_grad():
            # Try greedy first (no beam, no n-gram blocking).
            ids_greedy = model.generate(pixel_values, num_beams=1, do_sample=False,
                                        max_length=128).cpu()[0].tolist()
            # Beam search as configured by train_trocr.py.
            ids_beam = model.generate(pixel_values, num_beams=4, do_sample=False,
                                      max_length=128, early_stopping=True,
                                      no_repeat_ngram_size=3).cpu()[0].tolist()

        text_greedy = processor.batch_decode([ids_greedy])[0]
        text_beam = processor.batch_decode([ids_beam])[0]

        print(f"GT       : {gt}")
        print(f"Greedy   : {text_greedy!r}")
        print(f"  ids[:30]: {ids_greedy[:30]}")
        print(f"Beam=4   : {text_beam!r}")
        print(f"  ids[:30]: {ids_beam[:30]}")
        print("-" * 40)


if __name__ == "__main__":
    main()
