"""End-to-end Part B pipeline: image -> Part A YOLO -> class-1 crop -> CRNN -> digits.

Output line per image (per PDF spec — digits only, no '.' or '/'):
    <filename> <digits>

Optional --save-crops <dir> writes predicted-bbox crops to <dir>/<stem>.png so
they can be fed back into evaluate.py for pipeline-mode metrics.

Usage:
    python -m PartB.predict \\
        --partA C:\\Users\\qxawe\\NLP_Project\\.claude\\worktrees\\elated-meninsky-959dc4\\PartA\\runs\\yolov8s_1280-2\\weights\\best.pt \\
        --recognizer PartB/runs/crnn_b1/weights/best.pt \\
        --images C:\\Users\\qxawe\\NLP_Project\\Dataset\\CheckImages \\
        --out predictions_courtesy.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from ultralytics import YOLO

from PartB.courtesy_tokenizer import load_vocab, to_digits_only
from PartB.dataset import DEFAULT_HEIGHT
from PartB.decode import greedy_ctc_decode
from PartC.model import build_model

COURTESY_CLASS_ID = 1


def crop_with_margin(img: Image.Image, box, margin: int) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = img.size
    return img.crop((
        max(0, int(round(x1)) - margin),
        max(0, int(round(y1)) - margin),
        min(w, int(round(x2)) + margin),
        min(h, int(round(y2)) + margin),
    ))


def best_courtesy_box(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    xyxy_arr = boxes.xyxy.cpu().numpy()
    best = None
    for c, conf, xyxy in zip(cls_arr, conf_arr, xyxy_arr):
        if c != COURTESY_CLASS_ID:
            continue
        if best is None or conf > best[0]:
            best = (float(conf), tuple(map(float, xyxy)))
    return best[1] if best else None


@torch.no_grad()
def recognize(crop: Image.Image, model, idx_to_char, height, device) -> str:
    g = crop.convert("L")
    w, h = g.size
    new_w = max(1, int(round(w * height / h)))
    g = g.resize((new_w, height), Image.BILINEAR)
    tensor = TF.to_tensor(g).unsqueeze(0).to(device)
    log_probs = model(tensor)
    return greedy_ctc_decode(log_probs.cpu(), idx_to_char)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partA", type=Path, required=True)
    ap.add_argument("--recognizer", type=Path, required=True, help="PartB CRNN best.pt")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-crops", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Dir containing courtesy_vocab.json. Default: PartB")
    ap.add_argument("--ext", default=".tif")
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-nms", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    char_to_idx, idx_to_char, vocab_size = load_vocab(data_dir / "courtesy_vocab.json")

    ckpt = torch.load(args.recognizer, map_location="cpu", weights_only=False)
    height = ckpt.get("height", DEFAULT_HEIGHT)
    backbone = ckpt.get("backbone", "resnet18")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crnn = build_model(backbone, vocab_size=vocab_size).to(device)
    crnn.load_state_dict(ckpt["model"])
    crnn.eval()

    yolo = YOLO(str(args.partA))

    images = sorted(Path(args.images).glob(f"*{args.ext}"))
    if not images:
        raise SystemExit(f"No {args.ext} images in {args.images}")
    print(f"Predicting on {len(images)} images")

    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

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
            box = best_courtesy_box(result)
            if box is None:
                n_no_box += 1
                f.write(f"{img_path.name}\n")
                continue
            crop = crop_with_margin(im_rgb, box, args.margin)
            if args.save_crops:
                crop.convert("L").save(args.save_crops / f"{img_path.stem}.png")
            text = recognize(crop, crnn, idx_to_char, height, device)
            digits = to_digits_only(text)
            f.write(f"{img_path.name} {digits}\n")

    print(f"Wrote {args.out}. Missed courtesy detection: {n_no_box}/{len(images)}")
    if args.save_crops:
        print(f"Saved predicted crops under {args.save_crops}")


if __name__ == "__main__":
    main()
