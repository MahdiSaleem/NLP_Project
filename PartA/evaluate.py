"""Evaluate a trained YOLOv8 checkpoint with the assignment's exact metrics.

Computes for each class (legal, courtesy) and overall:
  - Accuracy@t = fraction of samples with IoU >= t, for t in {0.50, 0.75, 0.90}
  - Mean IoU

For each test image, the highest-confidence prediction per class is matched against
the ground-truth box. A class with no prediction contributes IoU=0 for that sample.

Usage (from repo root):
    python PartA/evaluate.py
    python PartA/evaluate.py --weights PartA/runs/yolov8s_checks/weights/best.pt
"""
import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


CLASS_NAMES = {0: "legal_amount", 1: "courtesy_amount"}
THRESHOLDS = (0.50, 0.75, 0.90)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path: Path, img_w: int, img_h: int) -> dict:
    """Returns {class_id: xyxy} for ground-truth boxes."""
    gt = {}
    if not label_path.is_file():
        return gt
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        gt[cls] = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
    return gt


def best_pred_per_class(result) -> dict:
    """Return {class_id: xyxy} keeping the highest-confidence box per class."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {}
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    xyxy_arr = boxes.xyxy.cpu().numpy()
    best = {}
    for cls_id, conf, xyxy in zip(cls_arr, conf_arr, xyxy_arr):
        if cls_id not in best or conf > best[cls_id][0]:
            best[cls_id] = (float(conf), tuple(map(float, xyxy)))
    return {c: b[1] for c, b in best.items()}


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(script_dir / "runs" / "yolov8s_checks" / "weights" / "best.pt"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--yolo-data", default=str(script_dir / "yolo_data"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-nms", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    yolo_data = Path(args.yolo_data)
    img_dir = yolo_data / "images" / args.split
    label_dir = yolo_data / "labels" / args.split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    if not images:
        raise SystemExit(f"No images found in {img_dir}")
    print(f"Evaluating {len(images)} images from split '{args.split}'")

    model = YOLO(args.weights)
    ious_per_class: dict[int, list[float]] = {c: [] for c in CLASS_NAMES}
    per_image_records = []

    for img_path in images:
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        gt = load_gt(label_dir / f"{img_path.stem}.txt", img_w, img_h)
        result = model.predict(
            source=str(img_path), conf=args.conf, iou=args.iou_nms,
            imgsz=args.imgsz, device=args.device, verbose=False, max_det=10,
        )[0]
        preds = best_pred_per_class(result)

        record = {"image": img_path.name, "iou": {}}
        for cls_id in CLASS_NAMES:
            if cls_id not in gt:
                continue  # no GT for this class on this image -> skip
            pred_box = preds.get(cls_id)
            score = iou(pred_box, gt[cls_id]) if pred_box is not None else 0.0
            ious_per_class[cls_id].append(score)
            record["iou"][CLASS_NAMES[cls_id]] = score
        per_image_records.append(record)

    # Aggregate.
    summary = {"per_class": {}, "overall": {}}
    all_ious = []
    for cls_id, name in CLASS_NAMES.items():
        ious = ious_per_class[cls_id]
        all_ious.extend(ious)
        if not ious:
            summary["per_class"][name] = {"n": 0}
            continue
        n = len(ious)
        summary["per_class"][name] = {
            "n": n,
            "mean_iou": sum(ious) / n,
            **{f"acc@{t:.2f}": sum(1 for x in ious if x >= t) / n for t in THRESHOLDS},
        }
    if all_ious:
        n = len(all_ious)
        summary["overall"] = {
            "n": n,
            "mean_iou": sum(all_ious) / n,
            **{f"acc@{t:.2f}": sum(1 for x in all_ious if x >= t) / n for t in THRESHOLDS},
        }

    # Print report.
    print(f"\n=== Results on '{args.split}' split ===")
    for scope, stats in [*summary["per_class"].items(), ("overall", summary["overall"])]:
        if not stats or stats.get("n", 0) == 0:
            continue
        line = f"{scope:>16}  n={stats['n']:>4}  meanIoU={stats['mean_iou']:.4f}"
        for t in THRESHOLDS:
            line += f"  acc@{t:.2f}={stats[f'acc@{t:.2f}']:.4f}"
        print(line)

    out_path = Path(args.out) if args.out else script_dir / f"results_{args.split}.json"
    out_path.write_text(json.dumps({"summary": summary, "per_image": per_image_records}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
