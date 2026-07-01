"""Generate the README hero image: run the trained model on a few real val frames,
overlay zones + violations, save assets/demo.png.

Run after training:  python scripts/make_demo.py --weights runs/ppe_yolo11s/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from ppe_studio.infer import PPEDetector  # noqa: E402
from ppe_studio.runtime import load_zones  # noqa: E402
from ppe_studio.viz import render  # noqa: E402


def pick_frames(val_dir: Path, n: int) -> list[Path]:
    """Prefer frames that trigger violations (a no-hardhat label present) for a vivid demo."""
    imgs = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))
    labels = val_dir.parent.parent / "labels" / "val"
    violators, others = [], []
    for img in imgs:
        lbl = labels / f"{img.stem}.txt"
        if lbl.exists() and any(line.startswith("1 ") for line in lbl.read_text().splitlines()):
            violators.append(img)
        else:
            others.append(img)
    chosen = violators[:n]
    return chosen + others[: max(0, n - len(chosen))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ppe_yolo11s/weights/best.pt")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="assets/demo.png")
    args = ap.parse_args()

    zones = load_zones(ROOT / "configs" / "zones.yaml")
    detector = PPEDetector(args.weights, ROOT / "configs" / "classes.yaml", zones=zones, conf=0.35)

    frames = pick_frames(ROOT / "data" / "hardhat" / "images" / "val", args.n)
    if not frames:
        raise SystemExit("no val frames found — run data_nokey first")

    rendered = [render(Image.open(f), detector.predict_image(f), zones) for f in frames]

    # Horizontal montage at a common height.
    h = min(im.height for im in rendered)
    scaled = [im.resize((int(im.width * h / im.height), h)) for im in rendered]
    total_w = sum(im.width for im in scaled) + 12 * (len(scaled) - 1)
    canvas = Image.new("RGB", (total_w, h), (247, 245, 241))
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width + 12

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
