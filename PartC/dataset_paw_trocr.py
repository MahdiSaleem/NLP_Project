"""TrOCR-PAW dataset: TrOCR image preproc + PAW-decoder-token labels."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PAWTrOCRDataset(Dataset):
    def __init__(self, records_path: Path, crops_dir: Path, processor,
                 max_label_len: int = 64) -> None:
        self.records = json.loads(Path(records_path).read_text(encoding="utf-8"))
        self.crops_dir = Path(crops_dir)
        self.processor = processor
        self.max_label_len = max_label_len
        self.records = [
            r for r in self.records if (self.crops_dir / f"{r['image_id']}.png").is_file()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(self.crops_dir / f"{rec['image_id']}.png").convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            rec["paws"],
            padding="max_length",
            max_length=self.max_label_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        pad_id = self.processor.tokenizer.pad_token_id
        labels = labels.clone()
        labels[labels == pad_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_id": rec["image_id"],
            "paws": rec["paws"],
        }


def collate_paw_trocr(batch: list[dict]) -> dict:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "image_ids": [item["image_id"] for item in batch],
        "paws": [item["paws"] for item in batch],
    }
