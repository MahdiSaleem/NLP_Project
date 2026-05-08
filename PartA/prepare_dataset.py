"""Prepare YOLOv8 dataset for Part A: split 80/10/10, build YOLO layout, write data.yaml.

Usage (from repo root):
    python PartA/prepare_dataset.py
    python PartA/prepare_dataset.py --dataset-dir /abs/path/to/Dataset
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image


CLASSES = {0: "legal_amount", 1: "courtesy_amount"}
SEED = 42
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train, val, test
IMG_EXT = ".png"  # converted, 3-channel


def find_default_dataset_dir(script_dir: Path) -> Path:
    """Look for Dataset/CheckImages next to the repo, then in known fallback locations."""
    candidates = [
        script_dir.parent / "Dataset",
        Path(r"C:\Users\qxawe\NLP_Project\Dataset"),
    ]
    for c in candidates:
        if (c / "CheckImages").is_dir() and (c / "BoundingBoxes").is_dir():
            return c
    raise SystemExit(
        "Could not locate Dataset/CheckImages and Dataset/BoundingBoxes. "
        "Pass --dataset-dir explicitly."
    )


def link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src to dst; if symlinks aren't permitted on Windows, copy instead."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def convert_to_rgb(src: Path, dst: Path) -> None:
    """Save src image as 3-channel RGB to dst. Skip if dst already exists."""
    if dst.exists():
        return
    with Image.open(src) as im:
        im.convert("RGB").save(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to materialize yolo_data/. Default: PartA/yolo_data")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = args.dataset_dir or find_default_dataset_dir(script_dir)
    images_dir = dataset_dir / "CheckImages"
    bbox_dir = dataset_dir / "BoundingBoxes"
    out_dir = (args.out_dir or (script_dir / "yolo_data")).resolve()
    splits_dir = script_dir / "splits"

    print(f"Dataset dir: {dataset_dir}")
    print(f"Output yolo_data dir: {out_dir}")

    # Pair each image with its label, drop unlabeled images.
    images = sorted(images_dir.glob("*.tif"))
    pairs = []
    skipped = []
    for img in images:
        label = bbox_dir / f"{img.stem}.txt"
        if label.is_file():
            pairs.append((img, label))
        else:
            skipped.append(img.name)
    print(f"Found {len(images)} images, {len(pairs)} with labels, skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped (no label): {skipped}")

    # Deterministic shuffle + split.
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }
    for name, items in splits.items():
        print(f"  {name}: {len(items)}")

    # Materialize YOLO layout.
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        print(f"  Materializing {split} ({len(items)})...")
        with open(splits_dir / f"{split}.txt", "w", encoding="utf-8") as f:
            for img, label in items:
                dst_img = out_dir / "images" / split / f"{img.stem}{IMG_EXT}"
                convert_to_rgb(img, dst_img)
                link_or_copy(label, out_dir / "labels" / split / label.name)
                f.write(f"{dst_img.name}\n")

    # Write data.yaml.
    data_yaml = script_dir / "data.yaml"
    names_block = "\n".join(f"  {k}: {v}" for k, v in CLASSES.items())
    data_yaml.write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"Wrote {data_yaml}")
    print(f"Wrote split lists to {splits_dir}")


if __name__ == "__main__":
    main()
