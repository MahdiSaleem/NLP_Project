"""End-to-end pipeline using fine-tuned TrOCR-PAW as the recognizer.

Image -> Part A (YOLO, class 0 = legal amount) -> crop -> TrOCR-PAW -> PAW list.
Output: per image one line `<filename> <space-joined PAWs>`.

Usage:
    python -m PartC.predict_trocr_paw \\
        --partA <yolo best.pt> --partC PartC/runs/trocr_paw/best \\
        --images Dataset/CheckImages \\
        --out predictions_legal_paw.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from PartC.evaluate_trocr_paw import load_model_and_processor

LEGAL_CLASS_ID = 0


def crop_with_margin(img: Image.Image, box, margin: int) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = img.size
    return img.crop((
        max(0, int(round(x1)) - margin),
        max(0, int(round(y1)) - margin),
        min(w, int(round(x2)) + margin),
        min(h, int(round(y2)) + margin),
    ))


def best_legal_box(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    xyxy_arr = boxes.xyxy.cpu().numpy()
    best = None
    for c, conf, xyxy in zip(cls_arr, conf_arr, xyxy_arr):
        if c != LEGAL_CLASS_ID:
            continue
        if best is None or conf > best[0]:
            best = (float(conf), tuple(map(float, xyxy)))
    return best[1] if best else None


@torch.no_grad()
def recognize(crop: Image.Image, model, processor, device, gen_kwargs: dict) -> list[str]:
    rgb = crop.convert("RGB")
    pixel_values = processor.image_processor(images=rgb, return_tensors="pt").pixel_values.to(device)
    generated = model.generate(pixel_values, **gen_kwargs)
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partA", type=Path, required=True)
    ap.add_argument("--partC", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-crops", type=Path, default=None)
    ap.add_argument("--ext", default=".tif")
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-nms", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(args.partC)
    model.to(device).eval()
    print(f"Loaded TrOCR-PAW from {args.partC}")

    yolo = YOLO(str(args.partA))

    images = sorted(Path(args.images).glob(f"*{args.ext}"))
    if not images:
        raise SystemExit(f"No {args.ext} images in {args.images}")
    print(f"Predicting on {len(images)} images")

    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gen_kwargs = dict(num_beams=args.num_beams, max_length=args.max_length,
                      early_stopping=True, no_repeat_ngram_size=0)
    n_no_box = 0
    with args.out.open("w", encoding="utf-8") as f:
        for img_path in images:
            with Image.open(img_path) as im_raw:
                im_rgb = im_raw.convert("RGB")
                arr = np.array(im_rgb)
            result = yolo.predict(
                source=arr[:, :, ::-1],
                conf=args.conf, iou=args.iou_nms, imgsz=args.imgsz,
                device=args.device, verbose=False, max_det=10,
            )[0]
            box = best_legal_box(result)
            if box is None:
                n_no_box += 1
                f.write(f"{img_path.name}\n")
                continue
            crop = crop_with_margin(im_rgb, box, args.margin)
            if args.save_crops:
                crop.convert("L").save(args.save_crops / f"{img_path.stem}.png")
            paws = recognize(crop, model, processor, device, gen_kwargs)
            f.write(f"{img_path.name} {' '.join(paws)}\n")

    print(f"Wrote {args.out}. Missed legal-amount detection: {n_no_box}/{len(images)}")


if __name__ == "__main__":
    main()
