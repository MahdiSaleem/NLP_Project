"""Prepare Part B data: crop courtesy-amount patches with GT bboxes (class 1),
build train/val/test splits + digit vocab.

Inputs (absolute paths if --dataset-dir not given):
    Dataset/CheckImages/<id>.tif         (1800 source check images)
    Dataset/BoundingBoxes/<id>.txt       (YOLO labels; class 1 = courtesy)
    Dataset/CourtesyAmounts/*.txt        (tokenized labels — canonical, 1799 entries)
    PartA/splits/{train,val,test}.txt    (image-stem split assignment)

Outputs:
    PartB/crops/gt/{train,val,test}/<id>.png   grayscale courtesy crops
    PartB/splits/{train,val,test}.json         list of {image_id, text, bbox_xyxy}
    PartB/courtesy_vocab.json                  char_to_idx + counts (blank=0)

Usage:
    python -m PartB.prepare_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from PartB.courtesy_tokenizer import (
    build_vocab,
    load_all_courtesy,
    save_vocab,
)

COURTESY_CLASS_ID = 1


def find_default_dataset_dir(script_dir: Path) -> Path:
    candidates = [
        script_dir.parent / "Dataset",
        Path(r"C:\Users\qxawe\NLP_Project\Dataset"),
    ]
    for c in candidates:
        if (c / "CheckImages").is_dir():
            return c
    raise SystemExit("Could not locate Dataset/. Pass --dataset-dir.")


def find_default_partA_splits(script_dir: Path) -> Path:
    candidates = [
        script_dir.parent / "PartA" / "splits",
        Path(r"C:\Users\qxawe\NLP_Project\.claude\worktrees\elated-meninsky-959dc4\PartA\splits"),
    ]
    for c in candidates:
        if (c / "train.txt").is_file():
            return c
    raise SystemExit("Could not locate PartA/splits/. Pass --partA-splits.")


def load_courtesy_bbox(bbox_path: Path, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if not bbox_path.is_file():
        return None
    with open(bbox_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, cx, cy, w, h = line.split()
            if int(cls) != COURTESY_CLASS_ID:
                continue
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            x1 = int(round((cx - w / 2) * img_w))
            y1 = int(round((cy - h / 2) * img_h))
            x2 = int(round((cx + w / 2) * img_w))
            y2 = int(round((cy + h / 2) * img_h))
            return x1, y1, x2, y2
    return None


def crop_with_margin(img: Image.Image, box: tuple[int, int, int, int], margin: int) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = img.size
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return img.crop((x1, y1, x2, y2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--partA-splits", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write crops + splits + vocab. Default: PartB")
    ap.add_argument("--margin", type=int, default=2,
                    help="Pixel margin around GT bbox (courtesy crops are tiny — 2 default)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = args.dataset_dir or find_default_dataset_dir(script_dir)
    partA_splits = args.partA_splits or find_default_partA_splits(script_dir)
    out_dir = args.out_dir or script_dir

    images_dir = dataset_dir / "CheckImages"
    bbox_dir = dataset_dir / "BoundingBoxes"
    label_dir = dataset_dir / "CourtesyAmounts"
    crops_root = out_dir / "crops" / "gt"
    splits_out = out_dir / "splits"
    splits_out.mkdir(parents=True, exist_ok=True)

    print(f"Dataset dir:    {dataset_dir}")
    print(f"PartA splits:   {partA_splits}")
    print(f"Output dir:     {out_dir}")

    labels = load_all_courtesy(label_dir)
    print(f"Tokenized courtesy labels parsed: {len(labels)}")

    splits: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        path = partA_splits / f"{name}.txt"
        stems = [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]
        splits[name] = stems
        print(f"  {name}: {len(stems)} stems from PartA")

    per_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    miss_text = miss_bbox = miss_image = 0

    for split, stems in splits.items():
        crop_dir = crops_root / split
        crop_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {split}...")
        for stem in stems:
            text = labels.get(stem)
            if not text:
                miss_text += 1
                continue
            img_path = images_dir / f"{stem}.tif"
            if not img_path.is_file():
                miss_image += 1
                continue
            with Image.open(img_path) as im:
                w, h = im.size
                box = load_courtesy_bbox(bbox_dir / f"{stem}.txt", w, h)
                if box is None:
                    miss_bbox += 1
                    continue
                crop = crop_with_margin(im, box, args.margin).convert("L")
                crop.save(crop_dir / f"{stem}.png")
            per_split[split].append({
                "image_id": stem,
                "text": text,
                "bbox_xyxy": list(box),
            })
        print(f"  {split}: {len(per_split[split])} usable")

    print(f"Skipped — missing text: {miss_text}, missing image: {miss_image}, missing bbox: {miss_bbox}")

    for split, recs in per_split.items():
        with open(splits_out / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)

    train_texts = [r["text"] for r in per_split["train"]]
    vocab = build_vocab(train_texts)
    save_vocab(vocab, out_dir / "courtesy_vocab.json")
    print(f"Vocab: {vocab['size'] - 1} symbols (+1 blank = {vocab['size']}). Counts: {vocab['char_counts']}")

    train_chars = set(vocab["char_to_idx"].keys())
    for split in ("val", "test"):
        oov = sum(1 for r in per_split[split] for c in r["text"] if c not in train_chars)
        total = sum(len(r["text"]) for r in per_split[split])
        if total:
            print(f"  {split}: {oov}/{total} chars OOV ({100*oov/total:.2f}%)")


if __name__ == "__main__":
    main()
