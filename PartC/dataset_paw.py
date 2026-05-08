"""PyTorch Dataset + collate for PAW-token CTC training (Part C v3).

Mirrors `dataset.py` for image preprocessing (height 96, variable width, white
pad collate) but emits PAW-ID labels instead of char-ID labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from PartC.paw_tokenizer import PAWTokenizer

DEFAULT_HEIGHT = 96


class PAWLineDataset(Dataset):
    def __init__(
        self,
        records_path: Path,
        crops_dir: Path,
        tokenizer: PAWTokenizer,
        height: int = DEFAULT_HEIGHT,
        augment: bool = False,
    ) -> None:
        self.records = json.loads(Path(records_path).read_text(encoding="utf-8"))
        self.crops_dir = Path(crops_dir)
        self.tokenizer = tokenizer
        self.height = height
        self.augment = augment
        self.records = [
            r for r in self.records if (self.crops_dir / f"{r['image_id']}.png").is_file()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-2, 2)
        translate = (random.randint(-2, 2), random.randint(-2, 2))
        shear = random.uniform(-2, 2)
        img = TF.affine(img, angle=angle, translate=translate, scale=1.0, shear=shear, fill=255)
        img = TF.adjust_brightness(img, 1.0 + random.uniform(-0.1, 0.1))
        img = TF.adjust_contrast(img, 1.0 + random.uniform(-0.1, 0.1))
        if random.random() < 0.5:
            t = TF.to_tensor(img)
            t = (t + torch.randn_like(t) * (random.uniform(0, 4) / 255.0)).clamp(0, 1)
            img = TF.to_pil_image(t)
        return img

    def _resize_keep_ratio(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = max(1, int(round(w * self.height / h)))
        return img.resize((new_w, self.height), Image.BILINEAR)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(self.crops_dir / f"{rec['image_id']}.png").convert("L")
        if self.augment:
            img = self._augment(img)
        img = self._resize_keep_ratio(img)
        tensor = TF.to_tensor(img)  # (1, H, W) in [0, 1]
        ids = self.tokenizer.encode(rec["paws"])
        return {
            "image": tensor,
            "label": torch.tensor(ids, dtype=torch.long),
            "label_length": len(ids),
            "width": tensor.shape[2],
            "image_id": rec["image_id"],
            "paws": rec["paws"],
        }


def collate_ctc_paw(batch: list[dict]) -> dict:
    heights = {item["image"].shape[1] for item in batch}
    assert len(heights) == 1, f"All images must share height, got {heights}"
    h = heights.pop()
    max_w = max(item["width"] for item in batch)
    b = len(batch)
    images = torch.ones(b, 1, h, max_w, dtype=torch.float32)
    widths = torch.zeros(b, dtype=torch.long)
    for i, item in enumerate(batch):
        w = item["width"]
        images[i, :, :, :w] = item["image"]
        widths[i] = w
    labels = torch.cat([item["label"] for item in batch]) if batch else torch.zeros(0, dtype=torch.long)
    label_lengths = torch.tensor([item["label_length"] for item in batch], dtype=torch.long)
    return {
        "images": images,
        "widths": widths,
        "labels": labels,
        "label_lengths": label_lengths,
        "image_ids": [item["image_id"] for item in batch],
        "paws": [item["paws"] for item in batch],
    }
