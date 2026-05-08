"""Build PAW splits + vocab. Run once before any PAW experiments.

Inputs:
    Dataset/LegalAmounts_tokenized/*.txt
    PartA/splits/{train,val,test}.txt           (image-stem split assignment)

Outputs:
    PartC/splits_paw/{train,val,test}.json       list of {image_id, paws: [...]}
    PartC/paw_vocab.json                          {train_paws: [...]} (sorted)
    PartC/crops/gt/{split}/<id>.png               REUSED from prepare_data.py — not regenerated
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from PartC.paw_tokenizer import PAWTokenizer, load_paw_dir
from PartC.prepare_data import crop_with_margin, load_legal_bbox


def find_default_dataset_dir(script_dir: Path) -> Path:
    candidates = [
        script_dir.parent / "Dataset",
        Path(r"C:\Users\qxawe\NLP_Project\Dataset"),
    ]
    for c in candidates:
        if (c / "LegalAmounts_tokenized").is_dir():
            return c
    raise SystemExit("Could not locate Dataset/LegalAmounts_tokenized.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--partA-splits", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    dataset_dir = args.dataset_dir or find_default_dataset_dir(script_dir)
    partA_splits = args.partA_splits or (repo_root / "PartA" / "splits")
    out_dir = args.out_dir or script_dir
    splits_out = out_dir / "splits_paw"
    splits_out.mkdir(parents=True, exist_ok=True)

    tok_dir = dataset_dir / "LegalAmounts_tokenized"
    print(f"Tokenized dir: {tok_dir}")
    print(f"PartA splits:  {partA_splits}")
    print(f"Output dir:    {out_dir}")

    paw_map = load_paw_dir(tok_dir)
    print(f"Tokenized labels: {len(paw_map)}")

    splits: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        path = partA_splits / f"{name}.txt"
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run PartA/prepare_dataset.py first.")
        stems = [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]
        splits[name] = stems

    per_split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing = {"train": 0, "val": 0, "test": 0}
    for split, stems in splits.items():
        for stem in stems:
            paws = paw_map.get(stem)
            if not paws:
                missing[split] += 1
                continue
            per_split_records[split].append({"image_id": stem, "paws": paws})
        print(f"  {split}: {len(per_split_records[split])} usable, missing {missing[split]}")

    # Build train vocab.
    paw_counts: Counter[str] = Counter()
    for rec in per_split_records["train"]:
        paw_counts.update(rec["paws"])
    train_paws = sorted(paw_counts.keys())
    print(f"Train PAW types: {len(train_paws)}, total tokens: {sum(paw_counts.values())}")

    tokenizer = PAWTokenizer(paws=train_paws)
    print(f"Vocab size (incl. blank+unk): {tokenizer.vocab_size}")

    # OOV stats.
    train_set = set(train_paws)
    for split in ("val", "test"):
        oov = sum(1 for rec in per_split_records[split] for p in rec["paws"] if p not in train_set)
        total = sum(len(rec["paws"]) for rec in per_split_records[split])
        if total:
            print(f"  {split}: {oov}/{total} PAWs OOV ({100*oov/total:.2f}%)")

    # Per-split sequence length stats.
    for split, recs in per_split_records.items():
        lens = [len(r["paws"]) for r in recs]
        if lens:
            lens_sorted = sorted(lens)
            mean = sum(lens) / len(lens)
            median = lens_sorted[len(lens) // 2]
            print(f"  {split}: PAW count min {min(lens)}, median {median}, mean {mean:.1f}, max {max(lens)}")

    # Write splits + vocab.
    for split, records in per_split_records.items():
        with open(splits_out / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    tokenizer.save(out_dir / "paw_vocab.json")

    # Crop any missing images (tokenized covers more IDs than raw, so a few records
    # may not yet have a GT crop on disk). We reuse the same crop logic prepare_data.py
    # uses; existing PNGs are skipped.
    crops_root = out_dir / "crops" / "gt"
    images_dir = dataset_dir / "CheckImages"
    bbox_dir = dataset_dir / "BoundingBoxes"
    cropped, skipped, failed = 0, 0, 0
    for split, recs in per_split_records.items():
        (crops_root / split).mkdir(parents=True, exist_ok=True)
        for rec in recs:
            dst = crops_root / split / f"{rec['image_id']}.png"
            if dst.is_file():
                skipped += 1
                continue
            img_path = images_dir / f"{rec['image_id']}.tif"
            if not img_path.is_file():
                failed += 1
                continue
            with Image.open(img_path) as im:
                w, h = im.size
                box = load_legal_bbox(bbox_dir / f"{rec['image_id']}.txt", w, h)
                if box is None:
                    failed += 1
                    continue
                im.crop(box)
                crop = crop_with_margin(im, box, 4).convert("L")
                crop.save(dst)
                cropped += 1
    print(f"Crops: existing {skipped}, newly cropped {cropped}, failed {failed}")
    # Drop any records that still lack a crop.
    for split in per_split_records:
        per_split_records[split] = [
            r for r in per_split_records[split]
            if (crops_root / split / f"{r['image_id']}.png").is_file()
        ]
    for split, recs in per_split_records.items():
        with open(splits_out / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"  {split} final: {len(recs)} records")


if __name__ == "__main__":
    main()
