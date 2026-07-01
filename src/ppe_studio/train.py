"""Fine-tune a YOLO model on the PPE dataset.

Defaults are tuned for a single RTX 3080 (10 GB). Long runs are meant to go overnight:
    python -m ppe_studio.train --epochs 100 --model yolo11s.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import resolve_dataset
from .runtime import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO for PPE detection.")
    ap.add_argument("--model", default="yolo11s.pt", help="base weights (yolo11s.pt / yolo12s.pt)")
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="lower to 8 if 10GB OOMs")
    ap.add_argument("--device", default="auto", help="auto picks the largest-memory GPU")
    ap.add_argument("--name", default="ppe_yolo11s")
    ap.add_argument("--fraction", type=float, default=1.0, help="dataset fraction (use <1 for smoke tests)")
    args = ap.parse_args()

    from ultralytics import YOLO

    # Absolute project path → predictable save_dir (a relative project nests under
    # Ultralytics' runs_dir/task and produces runs/detect/runs/<name>).
    repo_root = Path(__file__).resolve().parents[2]
    project = str(repo_root / "runs")

    data_yaml = resolve_dataset(args.dataset_config)
    device = resolve_device(args.device)
    print(f"[train] data={data_yaml} device={device} model={args.model}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=project,
        name=args.name,
        patience=20,
        fraction=args.fraction,
        plots=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"[train] done — best weights: {best}")


if __name__ == "__main__":
    main()
