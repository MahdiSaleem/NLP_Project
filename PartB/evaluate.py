"""Evaluate a Part B CRNN checkpoint (oracle GT crops or pipeline crops).

Usage (oracle, default):
    python -m PartB.evaluate --weights PartB/runs/crnn_b1/weights/best.pt --split test

Pipeline mode (use YOLO-predicted crops saved by predict.py):
    python -m PartB.evaluate --weights ... --split test --crops-dir PartB/crops/preds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from PartB.courtesy_tokenizer import load_vocab
from PartB.dataset import DEFAULT_HEIGHT, CourtesyDataset, collate_ctc
from PartB.decode import digit_metrics, greedy_ctc_decode, prefix_beam_decode
from PartC.model import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--crops-dir", type=Path, default=None,
                    help="Override crops dir (e.g. PartB/crops/preds for pipeline-mode).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--beam", type=int, default=0,
                    help="If >0, use prefix-beam decode instead of greedy.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    char_to_idx, idx_to_char, vocab_size = load_vocab(data_dir / "courtesy_vocab.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "resnet18")
    height = ckpt.get("height", DEFAULT_HEIGHT)
    model = build_model(backbone, vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    crops_root = args.crops_dir or (data_dir / "crops" / "gt" / args.split)
    # If --crops-dir was given as a parent, no extra split suffix needed.
    if args.crops_dir is None:
        crops_dir = crops_root
    else:
        crops_dir = Path(args.crops_dir)

    ds = CourtesyDataset(data_dir / "splits" / f"{args.split}.json",
                         crops_dir, char_to_idx, height=height, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_ctc)
    print(f"Evaluating {args.split}: {len(ds)} samples from {crops_dir} (backbone={backbone}, h={height})")

    preds_all: list[str] = []
    gts_all: list[str] = []
    ids_all: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            log_probs = model(images).cpu()
            if args.beam > 0:
                preds = prefix_beam_decode(log_probs, idx_to_char, blank=0, beam_size=args.beam)
            else:
                preds = greedy_ctc_decode(log_probs, idx_to_char)
            preds_all.extend(preds)
            gts_all.extend(batch["texts"])
            ids_all.extend(batch["image_ids"])

    metrics = digit_metrics(preds_all, gts_all)
    print(json.dumps(metrics, indent=2))

    if args.out:
        from PartB.courtesy_tokenizer import to_digits_only
        per_sample = [
            {
                "image_id": i,
                "gt": g,
                "pred": p,
                "gt_digits": to_digits_only(g),
                "pred_digits": to_digits_only(p),
            }
            for i, g, p in zip(ids_all, gts_all, preds_all)
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "weights": str(args.weights),
            "split": args.split,
            "crops_dir": str(crops_dir),
            "decode": "beam" if args.beam > 0 else "greedy",
            "beam_size": args.beam,
            "metrics": metrics,
            "samples": per_sample,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
