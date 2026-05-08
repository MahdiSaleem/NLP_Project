"""PyTorch Dataset + collate for handwritten Arabic line OCR (Part C).

- Loads grayscale crops produced by prepare_data.py
- Resizes to fixed height (default 96), preserving aspect ratio
- Train-time augmentation: small affine + brightness/contrast + light noise
- Collate pads variable widths to the batch max with white (255)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

DEFAULT_HEIGHT = 96


class LegalLineDataset(Dataset):
    def __init__(
        self,
        records_path: Path,
        crops_dir: Path,
        char_to_idx: dict[str, int],
        height: int = DEFAULT_HEIGHT,
        augment: bool = False,
    ) -> None:
        self.records = json.loads(Path(records_path).read_text(encoding="utf-8"))
        self.crops_dir = Path(crops_dir)
        self.char_to_idx = char_to_idx
        self.height = height
        self.augment = augment
        # Drop records whose crop file is missing (shouldn't happen post-prepare, but safe).
        self.records = [r for r in self.records if (self.crops_dir / f"{r['image_id']}.png").is_file()]

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, img: Image.Image) -> Image.Image:
        # Small affine: rotation +/-2 deg, translation +/-2 px, shear +/-2 deg.
        angle = random.uniform(-2, 2)
        max_dx, max_dy = 2, 2
        translate = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
        shear = random.uniform(-2, 2)
        img = TF.affine(img, angle=angle, translate=translate, scale=1.0, shear=shear, fill=255)
        # Brightness / contrast +/-10%.
        img = TF.adjust_brightness(img, 1.0 + random.uniform(-0.1, 0.1))
        img = TF.adjust_contrast(img, 1.0 + random.uniform(-0.1, 0.1))
        # Light Gaussian noise (sigma <= 4 on a 0-255 scale).
        if random.random() < 0.5:
            t = TF.to_tensor(img)
            t = (t + torch.randn_like(t) * (random.uniform(0, 4) / 255.0)).clamp(0, 1)
            img = TF.to_pil_image(t)
        return img

    def _resize_keep_ratio(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = max(1, int(round(w * self.height / h)))
        return img.resize((new_w, self.height), Image.BILINEAR)

    def encode(self, text: str) -> list[int]:
        # Train-only vocab + honest baseline: drop OOV chars (they count as deletions in eval).
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(self.crops_dir / f"{rec['image_id']}.png").convert("L")
        if self.augment:
            img = self._augment(img)
        img = self._resize_keep_ratio(img)
        # Normalize to [0, 1] then standardize lightly: invert is unnecessary, white bg is fine.
        tensor = TF.to_tensor(img)  # (1, H, W) in [0, 1]
        label = self.encode(rec["text"])
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "label_length": len(label),
            "width": tensor.shape[2],
            "image_id": rec["image_id"],
            "text": rec["text"],
        }


def collate_ctc(batch: list[dict]) -> dict:
    """Pad widths to max in batch with white (1.0 since to_tensor scaled to [0,1])."""
    heights = {item["image"].shape[1] for item in batch}
    assert len(heights) == 1, f"All images must share height, got {heights}"
    h = heights.pop()
    max_w = max(item["width"] for item in batch)
    b = len(batch)

    images = torch.ones(b, 1, h, max_w, dtype=torch.float32)  # white pad = 1.0
    widths = torch.zeros(b, dtype=torch.long)
    for i, item in enumerate(batch):
        w = item["width"]
        images[i, :, :, :w] = item["image"]
        widths[i] = w

    labels = torch.cat([item["label"] for item in batch]) if batch else torch.zeros(0, dtype=torch.long)
    label_lengths = torch.tensor([item["label_length"] for item in batch], dtype=torch.long)

    return {
        "images": images,
        "widths": widths,  # original widths (pre-pad), CNN downsample applied later
        "labels": labels,
        "label_lengths": label_lengths,
        "image_ids": [item["image_id"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


def load_vocab(vocab_path: Path) -> tuple[dict[str, int], dict[int, str], int]:
    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    char_to_idx = data["char_to_idx"]
    idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
    return char_to_idx, idx_to_char, data["size"]
