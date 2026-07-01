"""Evaluate a trained model and print an honest per-class mAP table.

The numbers this prints are the ONLY numbers allowed in the README. Never hand-edit them.
    python -m ppe_studio.eval --weights runs/ppe_yolo11s/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import resolve_dataset
from .runtime import resolve_device


def evaluate(weights: str, dataset_config: str, device: str = "auto") -> dict:
    from ultralytics import YOLO

    data_yaml = resolve_dataset(dataset_config)
    model = YOLO(weights)
    metrics = model.val(data=str(data_yaml), device=resolve_device(device), verbose=False)

    names = metrics.names if hasattr(metrics, "names") else model.names
    per_class = {}
    # Ultralytics exposes per-class AP50 via box.ap50 aligned with metrics.ap_class_index.
    try:
        ap50 = metrics.box.ap50
        ap = metrics.box.maps  # mAP50-95 per class
        for i, cidx in enumerate(metrics.box.ap_class_index):
            per_class[names[int(cidx)]] = {
                "mAP50": round(float(ap50[i]), 4),
                "mAP50_95": round(float(ap[int(cidx)]), 4),
            }
    except Exception as exc:  # noqa: BLE001 — surface but don't crash the summary
        per_class["_warning"] = f"per-class parse failed: {exc}"

    return {
        "weights": weights,
        "data": str(data_yaml),
        "overall": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PPE model, emit per-class mAP.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="reports/metrics.json")
    args = ap.parse_args()

    result = evaluate(args.weights, args.dataset_config, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    o = result["overall"]
    print(f"\nOverall  mAP50={o['mAP50']}  mAP50-95={o['mAP50_95']}  P={o['precision']}  R={o['recall']}")
    print("Per class (mAP50 / mAP50-95):")
    for cls, m in result["per_class"].items():
        if isinstance(m, dict) and "mAP50" in m:
            print(f"  {cls:<12} {m['mAP50']:.4f} / {m['mAP50_95']:.4f}")
    print(f"\nwritten → {args.out}  (paste these into the README — no hand-editing)")


if __name__ == "__main__":
    main()
