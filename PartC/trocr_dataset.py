"""PyTorch Dataset + collate for TrOCR fine-tuning on Part C legal-amount lines.

- Reads PNG crops from PartC/crops/{gt,preds}/{split}/<id>.png (grayscale -> RGB).
- Hands the image to the model's processor.image_processor (default 384x384 + ImageNet
  normalize). Aspect distortion is accepted; the pretrained model's robustness is the
  whole point of the transformer path.
- Tokenizes labels with processor.tokenizer; pad-token IDs become -100 so cross-entropy
  loss masks them.
- Collate returns pixel_values + labels tensors (no CTC-style width padding here).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_MAX_LABEL_LEN = 128


class TrOCRLegalLineDataset(Dataset):
    def __init__(
        self,
        records_path: Path,
        crops_dir: Path,
        processor,
        max_label_len: int = DEFAULT_MAX_LABEL_LEN,
    ) -> None:
        self.records = json.loads(Path(records_path).read_text(encoding="utf-8"))
        self.crops_dir = Path(crops_dir)
        self.processor = processor
        self.max_label_len = max_label_len
        # Drop records whose crop file is missing (defensive — shouldn't happen post-prepare).
        self.records = [r for r in self.records if (self.crops_dir / f"{r['image_id']}.png").is_file()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(self.crops_dir / f"{rec['image_id']}.png").convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            rec["text"],
            padding="max_length",
            max_length=self.max_label_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        # Mask pad tokens for the cross-entropy loss.
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels = labels.clone()
            labels[labels == pad_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_id": rec["image_id"],
            "text": rec["text"],
        }


def collate_trocr(batch: list[dict]) -> dict:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "image_ids": [item["image_id"] for item in batch],
        "texts": [item["text"] for item in batch],
    }
