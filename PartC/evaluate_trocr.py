"""Evaluate a fine-tuned TrOCR-CharVocab model on a Part C split.

Usage:
    python -m PartC.evaluate_trocr --model-dir PartC/runs/trocr_charvocab/best --split test
    # pipeline mode (predicted crops):
    python -m PartC.evaluate_trocr --model-dir ... --split test --crops-dir PartC/crops/preds
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


def load_model_and_processor(model_dir: Path):
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    char_tok = CharTokenizer.load(model_dir)
    processor = TrOCRCharProcessor(image_processor=image_processor, tokenizer=char_tok)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    # FIX: train_trocr.py resized the decoder but didn't update generation_config,
    # so the saved file has stale token IDs from the original microsoft/trocr-base-handwritten.
    # Force gen-config to match our char vocab.
    model.generation_config.pad_token_id = char_tok.pad_token_id
    model.generation_config.bos_token_id = char_tok.bos_token_id
    model.generation_config.eos_token_id = char_tok.eos_token_id
    model.generation_config.decoder_start_token_id = char_tok.bos_token_id
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--crops-dir", type=Path, default=None,
                    help="Crops root; expects <crops-dir>/<split>/<id>.png. Default: PartC/crops/gt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    crops_root = args.crops_dir or (data_dir / "crops" / "gt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load_model_and_processor(args.model_dir)
    model.to(device).eval()
    print(f"Loaded {args.model_dir} (vocab_size={processor.tokenizer.vocab_size})")

    ds = TrOCRLegalLineDataset(
        data_dir / "splits" / f"{args.split}.json",
        crops_root / args.split,
        processor,
        max_label_len=args.max_length,
    )
    print(f"Evaluating on {args.split}: {len(ds)} samples from {crops_root / args.split}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_trocr)

    preds_all: list[str] = []
    gts_all: list[str] = []
    ids_all: list[str] = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            generated = model.generate(
                pixel_values,
                num_beams=args.num_beams,
                max_length=args.max_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            preds_all.extend(processor.batch_decode(generated, skip_special_tokens=True))
            gts_all.extend(batch["texts"])
            ids_all.extend(batch["image_ids"])

    metrics = cer_wer(preds_all, gts_all)
    print(json.dumps(metrics, indent=2))

    if args.out:
        per_sample = [
            {"image_id": i, "gt": g, "pred": p}
            for i, g, p in zip(ids_all, gts_all, preds_all)
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "model_dir": str(args.model_dir),
            "split": args.split,
            "crops_dir": str(crops_root / args.split),
            "metrics": metrics,
            "samples": per_sample,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
