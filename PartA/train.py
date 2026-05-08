"""Train YOLOv8 on the Arabic check legal/courtesy detection task.

Usage (from repo root, after running prepare_dataset.py):
    python PartA/train.py
    python PartA/train.py --model yolov8m.pt --imgsz 1280 --epochs 80 --batch 4
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--data", default=str(script_dir / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", default="yolov8s_checks")
    parser.add_argument("--project", default=str(script_dir / "runs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
