"""Fine-tune microsoft/trocr-base-handwritten with PAW-token decoder vocab.

Differences from train_trocr.py (char vocab):
  - Tokenizer = PAWDecoderTokenizer (215 = 4 specials + 211 train PAWs).
  - Decoder is much shorter (mean ~14.5 tokens vs ~36 chars), so smaller
    max_label_len (64 default) and faster generation.
  - Eval reports PAW-WER + CER(joined) + CER(concat) via paw_metrics.

Critical fix vs the prior char run: generation_config is overwritten so the
saved checkpoint generates correctly.

Usage:
    python -m PartC.train_trocr_paw --name trocr_paw --epochs 30
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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from PartC.dataset_paw_trocr import PAWTrOCRDataset, collate_paw_trocr
from PartC.decode_paw import paw_metrics
from PartC.paw_trocr_adapter import PAWDecoderTokenizer, TrOCRPAWProcessor

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


def build_model_and_processor(model_id: str, paw_vocab_path: Path):
    base_processor = TrOCRProcessor.from_pretrained(model_id)
    paw_tok = PAWDecoderTokenizer.from_paw_vocab(paw_vocab_path)
    processor = TrOCRPAWProcessor(image_processor=base_processor.image_processor,
                                  tokenizer=paw_tok)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.decoder.resize_token_embeddings(paw_tok.vocab_size)
    # Update both config + generation_config so save_pretrained writes them.
    for cfg in [model.config, getattr(model.config, "decoder", None)]:
        if cfg is None:
            continue
        cfg.pad_token_id = paw_tok.pad_token_id
        cfg.bos_token_id = paw_tok.bos_token_id
        cfg.eos_token_id = paw_tok.eos_token_id
        cfg.vocab_size = paw_tok.vocab_size
    model.config.decoder_start_token_id = paw_tok.bos_token_id
    model.generation_config.pad_token_id = paw_tok.pad_token_id
    model.generation_config.bos_token_id = paw_tok.bos_token_id
    model.generation_config.eos_token_id = paw_tok.eos_token_id
    model.generation_config.decoder_start_token_id = paw_tok.bos_token_id
    return model, processor


@torch.no_grad()
def evaluate_split(model, processor, loader, device, gen_kwargs: dict) -> dict:
    model.eval()
    preds_all: list[list[str]] = []
    gts_all: list[list[str]] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        generated = model.generate(pixel_values, **gen_kwargs)
        preds_all.extend(processor.batch_decode(generated, skip_special_tokens=True))
        gts_all.extend(batch["paws"])
    return paw_metrics(preds_all, gts_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-id", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-label-len", type=int, default=64)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    splits_dir = data_dir / "splits_paw"
    crops_root = data_dir / "crops" / "gt"
    paw_vocab_json = data_dir / "paw_vocab.json"

    print(f"Loading + surgery: {args.model_id}")
    model, processor = build_model_and_processor(args.model_id, paw_vocab_json)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: total {n_params:,}")
    print(f"PAW decoder vocab size: {processor.tokenizer.vocab_size}")

    train_ds = PAWTrOCRDataset(splits_dir / "train.json", crops_root / "train",
                               processor, max_label_len=args.max_label_len)
    val_ds = PAWTrOCRDataset(splits_dir / "val.json", crops_root / "val",
                             processor, max_label_len=args.max_label_len)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_paw_trocr,
                              pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_paw_trocr,
                            pin_memory=pin)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_optim_steps = max(1, args.epochs * steps_per_epoch)

    out_dir = script_dir / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_csv = csv.writer(log_f)
    log_csv.writerow([
        "epoch", "lr", "train_loss",
        "val_paw_wer", "val_paw_cer", "val_raw_cer_concat", "val_exact_match", "secs",
    ])

    gen_kwargs = dict(
        num_beams=args.num_beams,
        max_length=args.max_label_len,
        early_stopping=True,
        no_repeat_ngram_size=0,  # PAWs DO repeat (e.g., "و" appears many times); don't block.
    )

    best_wer = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    optim_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_micro = 0
        t0 = time.time()
        optim.zero_grad(set_to_none=True)

        for micro_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / args.grad_accum

            if not torch.isfinite(loss):
                print(f"  WARN epoch {epoch} micro {micro_idx}: non-finite loss, skipping batch")
                optim.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum
            n_micro += 1

            if (micro_idx + 1) % args.grad_accum == 0:
                lr_scale = cosine_with_warmup(optim_step, total_optim_steps)
                for pg in optim.param_groups:
                    pg["lr"] = args.lr * lr_scale
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                optim_step += 1

        train_loss = epoch_loss / max(1, n_micro)
        val_metrics = evaluate_split(model, processor, val_loader, device, gen_kwargs)
        secs = time.time() - t0
        cur_lr = optim.param_groups[0]["lr"]
        print(
            f"epoch {epoch:2d}/{args.epochs}  "
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

        if val_metrics["paw_wer"] < best_wer:
            best_wer = val_metrics["paw_wer"]
            best_epoch = epoch
            epochs_since_improve = 0
            best_dir = out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save(best_dir)
            (best_dir / "metrics.json").write_text(
                json.dumps({"epoch": epoch, **val_metrics}, indent=2)
            )
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stop at epoch {epoch} (patience {args.patience}).")
                break

    log_f.close()

    last_dir = out_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)
    processor.save(last_dir)

    summary = {
        "best_epoch": best_epoch, "best_val_paw_wer": best_wer,
        "model_id": args.model_id, "epochs_run": epoch,
        "n_train": len(train_ds), "n_val": len(val_ds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best val PAW-WER {best_wer:.2f}% at epoch {best_epoch}. Wrote {out_dir}.")


if __name__ == "__main__":
    main()
