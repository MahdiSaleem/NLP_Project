"""Evaluate a fine-tuned TrOCR-PAW model on a Part C split.

Usage:
    python -m PartC.evaluate_trocr_paw --model-dir PartC/runs/trocr_paw/best --split test
    # pipeline mode:
    python -m PartC.evaluate_trocr_paw --model-dir ... --split test --crops-dir PartC/crops/preds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, VisionEncoderDecoderModel

from PartC.dataset_paw_trocr import PAWTrOCRDataset, collate_paw_trocr
from PartC.decode_paw import paw_metrics
from PartC.paw_trocr_adapter import PAWDecoderTokenizer, TrOCRPAWProcessor


def load_model_and_processor(model_dir: Path):
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    paw_tok = PAWDecoderTokenizer.load(model_dir)
    processor = TrOCRPAWProcessor(image_processor=image_processor, tokenizer=paw_tok)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    # Defensive: re-sync gen_config in case checkpoint was saved by an older version.
    model.generation_config.pad_token_id = paw_tok.pad_token_id
    model.generation_config.bos_token_id = paw_tok.bos_token_id
    model.generation_config.eos_token_id = paw_tok.eos_token_id
    model.generation_config.decoder_start_token_id = paw_tok.bos_token_id
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--crops-dir", type=Path, default=None,
                    help="Crops root; default: PartC/crops/gt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=64)
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

    ds = PAWTrOCRDataset(
        data_dir / "splits_paw" / f"{args.split}.json",
        crops_root / args.split,
        processor,
        max_label_len=args.max_length,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_paw_trocr)
    print(f"Evaluating on {args.split}: {len(ds)} samples from {crops_root / args.split}")

    preds_all: list[list[str]] = []
    gts_all: list[list[str]] = []
    ids_all: list[str] = []
    with torch.no_grad():
        for batch in loader:
            generated = model.generate(
                batch["pixel_values"].to(device),
                num_beams=args.num_beams,
                max_length=args.max_length,
                early_stopping=True,
                no_repeat_ngram_size=0,
            )
            preds_all.extend(processor.batch_decode(generated, skip_special_tokens=True))
            gts_all.extend(batch["paws"])
            ids_all.extend(batch["image_ids"])

    metrics = paw_metrics(preds_all, gts_all)
    print(json.dumps(metrics, indent=2))

    if args.out:
        per_sample = [
            {"image_id": i, "gt_paws": g, "pred_paws": p}
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
