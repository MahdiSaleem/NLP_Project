"""Evaluate a PAW-CTC checkpoint on a split.

Usage:
    python -m PartC.evaluate_paw --weights PartC/runs/paw_crnn/weights/best.pt --split test
    python -m PartC.evaluate_paw ... --beam 10               # beam search
    python -m PartC.evaluate_paw ... --crops-dir PartC/crops/preds   # pipeline mode
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from PartC.dataset_paw import PAWLineDataset, collate_ctc_paw
from PartC.decode_paw import beam_paw_decode, greedy_paw_decode, paw_metrics
from PartC.model import build_model
from PartC.paw_tokenizer import PAWTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--crops-dir", type=Path, default=None,
                    help="Crops root; default: PartC/crops/gt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--beam", type=int, default=0,
                    help="Beam width. 0 = greedy.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    crops_root = args.crops_dir or (data_dir / "crops" / "gt")

    tokenizer = PAWTokenizer.load(data_dir / "paw_vocab.json")
    vocab_size = tokenizer.vocab_size

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    if ckpt.get("vocab_size", vocab_size) != vocab_size:
        raise SystemExit(
            f"Vocab mismatch: weights expect {ckpt['vocab_size']}, paw_vocab.json has {vocab_size}."
        )
    backbone = ckpt.get("backbone", "resnet18")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone, vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Backbone: {backbone}, decode: {'beam=' + str(args.beam) if args.beam else 'greedy'}")

    height = ckpt.get("height", 96)
    ds = PAWLineDataset(
        data_dir / "splits_paw" / f"{args.split}.json",
        crops_root / args.split,
        tokenizer,
        height=height,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_ctc_paw)
    print(f"Evaluating on {args.split}: {len(ds)} samples from {crops_root / args.split}")

    preds_all: list[list[str]] = []
    gts_all: list[list[str]] = []
    ids_all: list[str] = []
    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["images"].to(device))
            if args.beam:
                preds_all.extend(beam_paw_decode(log_probs.cpu(), tokenizer, beam_size=args.beam))
            else:
                preds_all.extend(greedy_paw_decode(log_probs.cpu(), tokenizer))
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
            "split": args.split,
            "weights": str(args.weights),
            "decode": ("beam=" + str(args.beam)) if args.beam else "greedy",
            "crops_dir": str(crops_root / args.split),
            "metrics": metrics,
            "samples": per_sample,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
