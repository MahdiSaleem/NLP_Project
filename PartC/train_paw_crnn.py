"""CTC training of CRNN over PAW-token vocabulary (Part C v3, Exp 1).

Identical training scaffold to `train.py` but consumes PAW-tokens via
`PartC.dataset_paw` and reports PAW-WER + PAW-CER (joined-string char CER).

Usage:
    python -m PartC.train_paw_crnn --name paw_crnn --backbone resnet18
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from PartC.dataset_paw import DEFAULT_HEIGHT, PAWLineDataset, collate_ctc_paw
from PartC.decode_paw import greedy_paw_decode, paw_metrics
from PartC.model import build_model, count_params
from PartC.paw_tokenizer import PAWTokenizer

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(step: int, total: int, warmup_frac: float = 0.05) -> float:
    warmup_steps = max(1, int(total * warmup_frac))
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, tokenizer, device) -> dict:
    model.eval()
    preds_all: list[list[str]] = []
    gts_all: list[list[str]] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        log_probs = model(images)
        preds_all.extend(greedy_paw_decode(log_probs.cpu(), tokenizer))
        gts_all.extend(batch["paws"])
    return paw_metrics(preds_all, gts_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", choices=["vgg", "resnet18"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--clip-grad", type=float, default=5.0)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="PartC dir with splits_paw/ + crops/gt/ + paw_vocab.json. Default: PartC")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    splits_dir = data_dir / "splits_paw"
    crops_root = data_dir / "crops" / "gt"
    tokenizer = PAWTokenizer.load(data_dir / "paw_vocab.json")
    vocab_size = tokenizer.vocab_size  # incl. blank + unk
    print(f"PAW vocab size (incl. blank + unk): {vocab_size}")

    train_ds = PAWLineDataset(splits_dir / "train.json", crops_root / "train",
                              tokenizer, height=args.height, augment=True)
    val_ds = PAWLineDataset(splits_dir / "val.json", crops_root / "val",
                            tokenizer, height=args.height, augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_ctc_paw,
                              pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_ctc_paw,
                            pin_memory=pin)

    model = build_model(args.backbone, vocab_size=vocab_size).to(device)
    print(f"Backbone: {args.backbone}  params: {count_params(model):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    total_steps = max(1, args.epochs * len(train_loader))
    out_dir = script_dir / "runs" / args.name
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_csv = csv.writer(log_f)
    log_csv.writerow([
        "epoch", "lr", "train_loss",
        "val_paw_wer", "val_paw_cer", "val_raw_cer_concat", "val_exact_match", "secs",
    ])

    best_paw_wer = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            lr_scale = cosine_with_warmup(global_step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = args.lr * lr_scale

            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            label_lens = batch["label_lengths"].to(device, non_blocking=True)
            input_lens = model.output_lengths(batch["widths"]).to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                log_probs = model(images)
                loss = ctc(log_probs.float(), labels, input_lens, label_lens)

            if not torch.isfinite(loss):
                print(f"  WARN epoch {epoch} step {global_step}: non-finite loss, skipping")
                global_step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optim)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

        train_loss = epoch_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, tokenizer, device)
        secs = time.time() - t0
        cur_lr = optim.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"loss {train_loss:.4f}  "
            f"val WER {val_metrics['paw_wer']:.2f}  "
            f"CER(joined) {val_metrics['paw_cer_joined']:.2f}  "
            f"CER(concat) {val_metrics['raw_cer_concat']:.2f}  "
            f"exact {val_metrics['exact_match_rate']:.1f}%  "
            f"({secs:.1f}s, lr {cur_lr:.2e})"
        )
        log_csv.writerow([
            epoch, cur_lr, train_loss,
            val_metrics["paw_wer"], val_metrics["paw_cer_joined"],
            val_metrics["raw_cer_concat"], val_metrics["exact_match_rate"], secs,
        ])
        log_f.flush()

        if val_metrics["paw_wer"] < best_paw_wer:
            best_paw_wer = val_metrics["paw_wer"]
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_paw_wer": best_paw_wer,
                "vocab_size": vocab_size,
                "height": args.height,
                "backbone": args.backbone,
                "target": "paw",
            }, weights_dir / "best.pt")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stop at epoch {epoch} (patience {args.patience}).")
                break

    log_f.close()
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch, "vocab_size": vocab_size, "height": args.height,
        "backbone": args.backbone, "target": "paw",
    }, weights_dir / "last.pt")

    summary = {
        "best_epoch": best_epoch,
        "best_val_paw_wer": best_paw_wer,
        "epochs_run": epoch,
        "vocab_size": vocab_size,
        "backbone": args.backbone,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best val PAW-WER {best_paw_wer:.2f}% at epoch {best_epoch}. Wrote {out_dir}.")


if __name__ == "__main__":
    main()
