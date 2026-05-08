"""Evaluate a CRNN checkpoint on a split: greedy decode + CER/WER.

Two evaluation modes:
  - Oracle (recognizer-only):  --crops-dir PartC/crops/gt  --split test
  - Pipeline (end-to-end):     --crops-dir PartC/crops/preds  --split test
    (run PartC/predict.py first to materialize the predicted-crop directory)

Usage:
    python PartC/evaluate.py --weights PartC/runs/baseline/weights/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from PartC.dataset import LegalLineDataset, collate_ctc, load_vocab
from PartC.decode import cer_wer, greedy_ctc_decode
from PartC.model import CRNN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Dir with splits/ + vocab.json. Default: PartC")
    ap.add_argument("--crops-dir", type=Path, default=None,
                    help="Root dir of crops; expects <crops-dir>/<split>/<id>.png. "
                         "Default: PartC/crops/gt (oracle eval)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    crops_root = args.crops_dir or (data_dir / "crops" / "gt")

    char_to_idx, idx_to_char, vocab_size = load_vocab(data_dir / "vocab.json")

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    height = ckpt.get("height", 96)
    if ckpt.get("vocab_size", vocab_size) != vocab_size:
        raise SystemExit(
            f"Vocab mismatch: weights expect {ckpt['vocab_size']}, vocab.json has {vocab_size}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = LegalLineDataset(
        data_dir / "splits" / f"{args.split}.json",
        crops_root / args.split,
        char_to_idx,
        height=height,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_ctc)
    print(f"Evaluating on {args.split}: {len(ds)} samples from {crops_root / args.split}")

    preds_all: list[str] = []
    gts_all: list[str] = []
    ids_all: list[str] = []
    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["images"].to(device))
            preds_all.extend(greedy_ctc_decode(log_probs.cpu(), idx_to_char))
            gts_all.extend(batch["texts"])
            ids_all.extend(batch["image_ids"])

    metrics = cer_wer(preds_all, gts_all)
    print(json.dumps(metrics, indent=2))

    if args.out:
        per_sample = [
            {"image_id": i, "gt": g, "pred": p}
            for i, g, p in zip(ids_all, gts_all, preds_all)
        ]
        args.out.write_text(json.dumps({
            "split": args.split,
            "weights": str(args.weights),
            "crops_dir": str(crops_root / args.split),
            "metrics": metrics,
            "samples": per_sample,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
