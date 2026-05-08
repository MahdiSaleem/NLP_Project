"""Decision-gate zero-shot eval for a TrOCR-style HF model on a Part C split.

No training. Loads `<model-id>`, runs `.generate()` on the chosen split's GT crops,
reports CER/WER via PartC.decode.cer_wer.

Usage:
    python -m PartC.zero_shot_trocr --model-id RayR1/trocr-base-arabic-handwritten --split val
    python -m PartC.zero_shot_trocr --model-id microsoft/trocr-base-handwritten --split val --num-samples 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from PartC.decode import cer_wer
from PartC.trocr_dataset import TrOCRLegalLineDataset, collate_trocr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HF model ID, e.g. RayR1/trocr-base-arabic-handwritten")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="PartC dir with splits/ + crops/. Default: PartC")
    ap.add_argument("--crops-dir", type=Path, default=None,
                    help="Crops root; expects <crops-dir>/<split>/<id>.png. Default: PartC/crops/gt")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Cap samples for a quick smoke test")
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
    print(f"Device: {device}")

    print(f"Loading processor and model: {args.model_id}")
    processor = TrOCRProcessor.from_pretrained(args.model_id)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    ds = TrOCRLegalLineDataset(
        data_dir / "splits" / f"{args.split}.json",
        crops_root / args.split,
        processor,
    )
    if args.num_samples:
        ds.records = ds.records[: args.num_samples]
    print(f"Evaluating on {args.split}: {len(ds)} samples from {crops_root / args.split}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_trocr)

    preds_all: list[str] = []
    gts_all: list[str] = []
    ids_all: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(
                pixel_values,
                num_beams=args.num_beams,
                max_length=args.max_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            preds_all.extend(decoded)
            gts_all.extend(batch["texts"])
            ids_all.extend(batch["image_ids"])
    elapsed = time.time() - t0

    metrics = cer_wer(preds_all, gts_all)
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / max(1, len(ds)):.2f}s/sample)")
    print(json.dumps(metrics, indent=2))

    # First few samples for eyeballing.
    print("\nFirst 5 samples:")
    for i in range(min(5, len(preds_all))):
        print(f"  GT  : {gts_all[i]}")
        print(f"  PRED: {preds_all[i]}")
        print("  ---")

    if args.out:
        per_sample = [
            {"image_id": i, "gt": g, "pred": p}
            for i, g, p in zip(ids_all, gts_all, preds_all)
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "model_id": args.model_id,
            "split": args.split,
            "crops_dir": str(crops_root / args.split),
            "n_samples": len(preds_all),
            "elapsed_sec": elapsed,
            "metrics": metrics,
            "samples": per_sample,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
