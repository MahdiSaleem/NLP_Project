"""Prepare Part C data: crop legal-amount patches with GT bboxes, build splits + vocab.

Inputs:
    Dataset/CheckImages/<id>.tif         (1800 source check images)
    Dataset/BoundingBoxes/<id>.txt       (YOLO-format: class cx cy w h normalized)
    Dataset/LegalAmounts_raw_text/*.txt  (batched: <lineno>\\tL<id>.tif\\t<text>)
    PartA/splits/{train,val,test}.txt    (image-stem split assignment, .png filenames)

Outputs:
    PartC/crops/gt/{train,val,test}/<id>.png   grayscale legal-amount crops
    PartC/splits/{train,val,test}.json         list of {image_id, text, bbox_xyxy}
    PartC/vocab.json                           char_to_idx, idx_to_char (blank=0)
    PartC/vocab.txt                            human-readable char list

Usage (from repo root):
    python PartC/prepare_data.py
    python PartC/prepare_data.py --dataset-dir /abs/path/to/Dataset --margin 4
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

LEGAL_CLASS_ID = 0
RTL_LTR_CHARS = ["‫", "‬", "‎", "‏", "‪", "‭", "‮"]


def find_default_dataset_dir(script_dir: Path) -> Path:
    candidates = [
        script_dir.parent / "Dataset",
        Path(r"C:\Users\qxawe\NLP_Project\Dataset"),
    ]
    for c in candidates:
        if (c / "CheckImages").is_dir():
            return c
    raise SystemExit("Could not locate Dataset/. Pass --dataset-dir explicitly.")


def strip_bidi(text: str) -> str:
    for c in RTL_LTR_CHARS:
        text = text.replace(c, "")
    return text.strip()


def load_raw_text(raw_dir: Path) -> dict[str, str]:
    """Parse all batched raw-text files into {image_id: text}.

    Each line is `<id>.tif\\t<text>` with optional `L` prefix marking a Legal entry.
    """
    out: dict[str, str] = {}
    for fp in sorted(raw_dir.glob("*.txt")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                raw_id, text = parts
                if raw_id.startswith("L"):
                    raw_id = raw_id[1:]
                image_id = Path(raw_id).stem
                text = strip_bidi(text)
                if not text:
                    continue
                out[image_id] = text
    return out


def load_legal_bbox(bbox_path: Path, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Read YOLO label, return absolute pixel xyxy for the legal-amount class, or None."""
    if not bbox_path.is_file():
        return None
    with open(bbox_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, cx, cy, w, h = line.split()
            if int(cls) != LEGAL_CLASS_ID:
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
    ap.add_argument("--partA-splits", type=Path, default=None,
                    help="Dir with train.txt/val.txt/test.txt (image stems). Default: PartA/splits")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write crops + splits + vocab. Default: PartC")
    ap.add_argument("--margin", type=int, default=4, help="Pixel margin around GT bbox")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    dataset_dir = args.dataset_dir or find_default_dataset_dir(script_dir)
    partA_splits = args.partA_splits or (repo_root / "PartA" / "splits")
    out_dir = args.out_dir or script_dir

    images_dir = dataset_dir / "CheckImages"
    bbox_dir = dataset_dir / "BoundingBoxes"
    raw_text_dir = dataset_dir / "LegalAmounts_raw_text"
    crops_root = out_dir / "crops" / "gt"
    splits_out = out_dir / "splits"
    splits_out.mkdir(parents=True, exist_ok=True)

    print(f"Dataset dir:    {dataset_dir}")
    print(f"PartA splits:   {partA_splits}")
    print(f"Output dir:     {out_dir}")

    # Load all raw-text labels indexed by image stem.
    raw_texts = load_raw_text(raw_text_dir)
    print(f"Raw text labels: {len(raw_texts)}")

    # Load PartA splits — these list .png filenames (stems are what we want).
    splits: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        path = partA_splits / f"{name}.txt"
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run `python PartA/prepare_dataset.py` first.")
        stems = [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]
        splits[name] = stems
        print(f"  {name}: {len(stems)} stems from PartA")

    # Build per-split records, generate crops.
    per_split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing_text = 0
    missing_bbox = 0
    missing_image = 0

    for split, stems in splits.items():
        crop_dir = crops_root / split
        crop_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {split}...")
        for stem in stems:
            text = raw_texts.get(stem)
            if not text:
                missing_text += 1
                continue
            img_path = images_dir / f"{stem}.tif"
            if not img_path.is_file():
                missing_image += 1
                continue
            with Image.open(img_path) as im:
                w, h = im.size
                box = load_legal_bbox(bbox_dir / f"{stem}.txt", w, h)
                if box is None:
                    missing_bbox += 1
                    continue
                crop = crop_with_margin(im, box, args.margin).convert("L")
                crop.save(crop_dir / f"{stem}.png")
            per_split_records[split].append({
                "image_id": stem,
                "text": text,
                "bbox_xyxy": list(box),
            })
        print(f"  {split}: {len(per_split_records[split])} usable")

    print(f"Skipped — missing text: {missing_text}, missing image: {missing_image}, missing bbox: {missing_bbox}")

    # Persist splits.
    for split, records in per_split_records.items():
        with open(splits_out / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    # Build vocab from train texts only (honest baseline). Index 0 reserved for CTC blank.
    char_counts: Counter[str] = Counter()
    for rec in per_split_records["train"]:
        char_counts.update(rec["text"])
    chars = sorted(char_counts.keys())
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 is blank
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    vocab = {
        "blank": 0,
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "size": len(chars) + 1,  # incl. blank
    }
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(out_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for c in chars:
            f.write(f"{c}\t{char_counts[c]}\n")
    print(f"Vocab: {len(chars)} chars (+1 blank = {vocab['size']}). Wrote vocab.json + vocab.txt")

    # Stat: how many val/test chars are OOV? (counted as deletions per honest-baseline rule.)
    train_chars = set(chars)
    for split in ("val", "test"):
        oov = sum(1 for rec in per_split_records[split] for c in rec["text"] if c not in train_chars)
        total = sum(len(rec["text"]) for rec in per_split_records[split])
        if total:
            print(f"  {split}: {oov}/{total} chars OOV ({100*oov/total:.2f}%)")


if __name__ == "__main__":
    main()
