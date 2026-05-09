"""Train CRNN+CTC for Part B (courtesy-amount digit recognition).

Mirror of PartC/train_paw_crnn.py with PartB dataset, decode, vocab.

Usage:
    python -m PartB.train_crnn --name crnn_b1 --backbone resnet18 --epochs 100
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

from PartB.courtesy_tokenizer import load_vocab
from PartB.dataset import DEFAULT_HEIGHT, CourtesyDataset, collate_ctc
from PartB.decode import digit_metrics, greedy_ctc_decode
from PartC.model import build_model, count_params

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
def evaluate(model, loader, idx_to_char, device) -> dict:
    model.eval()
    preds_all: list[str] = []
    gts_all: list[str] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        log_probs = model(images)
        preds_all.extend(greedy_ctc_decode(log_probs.cpu(), idx_to_char))
        gts_all.extend(batch["texts"])
    return digit_metrics(preds_all, gts_all)


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
                    help="Dir with splits/, crops/gt/, courtesy_vocab.json. Default: PartB")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    splits_dir = data_dir / "splits"
    crops_root = data_dir / "crops" / "gt"
    char_to_idx, idx_to_char, vocab_size = load_vocab(data_dir / "courtesy_vocab.json")
    print(f"Vocab size (incl. blank): {vocab_size}  symbols: {sorted(char_to_idx.keys())}")

    train_ds = CourtesyDataset(splits_dir / "train.json", crops_root / "train",
                               char_to_idx, height=args.height, augment=True)
    val_ds = CourtesyDataset(splits_dir / "val.json", crops_root / "val",
                             char_to_idx, height=args.height, augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_ctc,
                              pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_ctc,
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
        "val_digit_acc", "val_pct_no_err", "val_pct_1_err", "val_pct_2plus_err", "secs",
    ])

    best_acc = -1.0
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
                print(f"  WARN epoch {epoch} step {global_step}: non-finite loss")
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
        m = evaluate(model, val_loader, idx_to_char, device)
        secs = time.time() - t0
        cur_lr = optim.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"loss {train_loss:.4f}  "
            f"acc {m['digit_accuracy']:.2f}  "
            f"no-err {m['pct_no_error']:.1f}%  "
            f"1-err {m['pct_one_error']:.1f}%  "
            f"2+err {m['pct_two_or_more_errors']:.1f}%  "
            f"({secs:.1f}s, lr {cur_lr:.2e})"
        )
        log_csv.writerow([
            epoch, cur_lr, train_loss,
            m["digit_accuracy"], m["pct_no_error"], m["pct_one_error"],
            m["pct_two_or_more_errors"], secs,
        ])
        log_f.flush()

        if m["digit_accuracy"] > best_acc:
            best_acc = m["digit_accuracy"]
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_digit_accuracy": best_acc,
                "vocab_size": vocab_size,
                "char_to_idx": char_to_idx,
                "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
                "height": args.height,
                "backbone": args.backbone,
            }, weights_dir / "best.pt")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stop at epoch {epoch} (patience {args.patience}).")
                break

    log_f.close()
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch, "vocab_size": vocab_size,
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "height": args.height, "backbone": args.backbone,
    }, weights_dir / "last.pt")

    summary = {
        "best_epoch": best_epoch,
        "best_val_digit_accuracy": best_acc,
        "epochs_run": epoch,
        "vocab_size": vocab_size,
        "backbone": args.backbone,
        "height": args.height,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best val digit accuracy {best_acc:.2f}% at epoch {best_epoch}. Wrote {out_dir}.")


if __name__ == "__main__":
    main()
