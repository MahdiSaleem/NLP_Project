"""Fine-tune microsoft/trocr-base-handwritten with a char-level Arabic tokenizer.

Architecture surgery:
  - Load the full VisionEncoderDecoderModel + processor from `--model-id`.
  - Replace the tokenizer with a CharTokenizer built from PartC/vocab.json.
  - Resize the decoder's token embeddings + LM head to the new vocab size.
  - Update model.config: pad_token_id, bos_token_id, eos_token_id,
    decoder_start_token_id, vocab_size.

Training: AdamW lr=5e-5, cosine schedule with 5% warmup, weight_decay=0.01.
Batch 4 with grad-accum 4 (effective 16). fp16. Patience-5 early stop on val CER.

Usage:
    python -m PartC.train_trocr --name trocr_charvocab
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

from PartC.char_tokenizer import CharTokenizer, TrOCRCharProcessor
from PartC.decode import cer_wer
from PartC.trocr_dataset import TrOCRLegalLineDataset, collate_trocr

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


def build_model_and_processor(model_id: str, vocab_json_path: Path):
    """Load TrOCR, swap tokenizer to char-level, resize decoder."""
    base_processor = TrOCRProcessor.from_pretrained(model_id)
    char_tok = CharTokenizer.from_partc_vocab(vocab_json_path)
    processor = TrOCRCharProcessor(image_processor=base_processor.image_processor,
                                   tokenizer=char_tok)

    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Resize decoder embeddings + LM head to char-vocab size.
    # `resize_token_embeddings` on the inner decoder is the cleanest way; HF's
    # VisionEncoderDecoderModel.resize_token_embeddings only affects the encoder
    # in some versions, so go through the decoder explicitly.
    model.decoder.resize_token_embeddings(char_tok.vocab_size)

    # Update config so generation uses our specials.
    model.config.pad_token_id = char_tok.pad_token_id
    model.config.bos_token_id = char_tok.bos_token_id
    model.config.eos_token_id = char_tok.eos_token_id
    model.config.decoder_start_token_id = char_tok.bos_token_id
    model.config.vocab_size = char_tok.vocab_size
    if hasattr(model.config, "decoder"):
        model.config.decoder.pad_token_id = char_tok.pad_token_id
        model.config.decoder.bos_token_id = char_tok.bos_token_id
        model.config.decoder.eos_token_id = char_tok.eos_token_id
        model.config.decoder.vocab_size = char_tok.vocab_size
    # Critical: also overwrite generation_config or save_pretrained writes the
    # original (microsoft/trocr-base-handwritten) values, which mismatch our vocab
    # and cause `generate()` to start with the wrong token (silent failure!).
    model.generation_config.pad_token_id = char_tok.pad_token_id
    model.generation_config.bos_token_id = char_tok.bos_token_id
    model.generation_config.eos_token_id = char_tok.eos_token_id
    model.generation_config.decoder_start_token_id = char_tok.bos_token_id

    return model, processor


@torch.no_grad()
def evaluate_split(model, processor, loader, device, gen_kwargs: dict) -> dict:
    model.eval()
    preds_all: list[str] = []
    gts_all: list[str] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        generated = model.generate(pixel_values, **gen_kwargs)
        preds_all.extend(processor.batch_decode(generated, skip_special_tokens=True))
        gts_all.extend(batch["texts"])
    return cer_wer(preds_all, gts_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Run name; output PartC/runs/<name>/")
    ap.add_argument("--model-id", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-label-len", type=int, default=128)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="PartC dir with splits/ + crops/gt/ + vocab.json. Default: PartC")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Enable on the encoder if VRAM tight.")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    splits_dir = data_dir / "splits"
    crops_root = data_dir / "crops" / "gt"
    vocab_json = data_dir / "vocab.json"

    print(f"Loading + surgery: {args.model_id}")
    model, processor = build_model_and_processor(args.model_id, vocab_json)
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        if hasattr(model, "decoder") and hasattr(model.decoder, "gradient_checkpointing_enable"):
            model.decoder.gradient_checkpointing_enable()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total {n_params:,}  trainable {n_trainable:,}")
    print(f"Char vocab size: {processor.tokenizer.vocab_size}")

    train_ds = TrOCRLegalLineDataset(splits_dir / "train.json", crops_root / "train",
                                     processor, max_label_len=args.max_label_len)
    val_ds = TrOCRLegalLineDataset(splits_dir / "val.json", crops_root / "val",
                                   processor, max_label_len=args.max_label_len)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_trocr,
                              pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_trocr,
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
    log_csv.writerow(["epoch", "lr", "train_loss", "val_cer", "val_wer", "val_exact_match", "secs"])

    gen_kwargs = dict(
        num_beams=args.num_beams,
        max_length=args.max_label_len,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    best_cer = float("inf")
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
            f"val CER {val_metrics['cer']:.2f}  WER {val_metrics['wer']:.2f}  "
            f"exact {val_metrics['exact_match_rate']:.1f}%  "
            f"({secs:.1f}s, lr {cur_lr:.2e})"
        )
        log_csv.writerow([
            epoch, cur_lr, train_loss,
            val_metrics["cer"], val_metrics["wer"], val_metrics["exact_match_rate"],
            secs,
        ])
        log_f.flush()

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
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
        "best_epoch": best_epoch,
        "best_val_cer": best_cer,
        "model_id": args.model_id,
        "epochs_run": epoch,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best val CER {best_cer:.2f}% at epoch {best_epoch}. Wrote {out_dir}.")


if __name__ == "__main__":
    main()
