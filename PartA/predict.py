"""Run a trained YOLOv8 checkpoint over a directory of check images.

Output format (one line per image), as specified in the assignment:
    {filename} {courtesy: x1 y1 x2 y2} {legal: x1 y1 x2 y2}

Coordinates are absolute pixel xyxy. If a class is missing, "NA" is written for its 4 fields.

Usage (from repo root):
    python PartA/predict.py --weights PartA/runs/yolov8s_checks/weights/best.pt \
                            --images Dataset/CheckImages --out predictions.txt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


LEGAL = 0
COURTESY = 1


def best_pred_per_class(result) -> dict:
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


def fmt_box(box) -> str:
    if box is None:
        return "NA NA NA NA"
    return " ".join(f"{int(round(v))}" for v in box)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True, help="Directory of check images")
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-nms", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--ext", default=".tif")
    args = parser.parse_args()

    images = sorted(Path(args.images).glob(f"*{args.ext}"))
    if not images:
        raise SystemExit(f"No {args.ext} images in {args.images}")
    print(f"Predicting on {len(images)} images")

    model = YOLO(args.weights)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for img in images:
            result = model.predict(
                source=str(img), conf=args.conf, iou=args.iou_nms,
                imgsz=args.imgsz, device=args.device, verbose=False, max_det=10,
            )[0]
            preds = best_pred_per_class(result)
            f.write(f"{img.name} {fmt_box(preds.get(COURTESY))} {fmt_box(preds.get(LEGAL))}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
