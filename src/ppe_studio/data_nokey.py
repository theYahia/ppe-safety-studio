"""No-key dataset preparation: keremberke/hard-hat-detection (Hugging Face) → YOLO layout.

Public CC-licensed Roboflow export of Hard Hat Workers, COCO format, 2 classes
(hardhat / no-hardhat). Downloaded via huggingface_hub (no API key, no Roboflow account),
then converted to the on-disk YOLO layout Ultralytics expects:

    <out>/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt
      data.yaml

Run:  python -m ppe_studio.data_nokey --out data/hardhat
Then set `local_path: data/hardhat` in configs/dataset.yaml (the script prints the path).
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

REPO_ID = "keremberke/hard-hat-detection"
# HF split name -> YOLO split dir
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def _download_zip(split: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=REPO_ID, filename=f"data/{split}.zip", repo_type="dataset")
    )


def _coco_to_yolo(coco: dict) -> tuple[dict[int, list[str]], list[str]]:
    """Return (image_id -> [yolo label lines], ordered class names).

    COCO bbox is [x_min, y_min, w, h] in pixels. YOLO is [cls cx cy w h] normalized.
    Category ids are remapped to a contiguous 0..N-1 index, names ordered to match.
    """
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    img_wh = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in coco["images"]}
    labels: dict[int, list[str]] = {im_id: [] for im_id in img_wh}

    for ann in coco["annotations"]:
        w_img, h_img, _ = img_wh[ann["image_id"]]
        x, y, bw, bh = ann["bbox"]
        cx = (x + bw / 2) / w_img
        cy = (y + bh / 2) / h_img
        nw, nh = bw / w_img, bh / h_img
        # clamp to [0,1] — guards against a few out-of-frame Roboflow boxes
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
        cls = id_to_idx[ann["category_id"]]
        labels[ann["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return labels, names


def prepare(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    names: list[str] = []
    counts: dict[str, int] = {}

    for hf_split, yolo_split in SPLITS.items():
        zip_path = _download_zip(hf_split)
        img_dir = out / "images" / yolo_split
        lbl_dir = out / "labels" / yolo_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as z:
            coco = json.loads(z.read("_annotations.coco.json"))
            labels, split_names = _coco_to_yolo(coco)
            names = names or split_names
            id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}

            for im_id, fname in id_to_file.items():
                with z.open(fname) as src:
                    (img_dir / fname).write_bytes(src.read())
                stem = Path(fname).stem
                (lbl_dir / f"{stem}.txt").write_text(
                    "\n".join(labels.get(im_id, [])), encoding="utf-8"
                )
        counts[yolo_split] = len(id_to_file)
        print(f"[data] {yolo_split}: {counts[yolo_split]} images")

    data_yaml = out / "data.yaml"
    lines = [
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        f"names: {names}",
    ]
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[data] classes: {names}")
    print(f"[data] wrote {data_yaml}")
    return data_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Hard Hat dataset (no API key).")
    ap.add_argument("--out", default="data/hardhat")
    args = ap.parse_args()
    path = prepare(args.out)
    print(f"\nSet this in configs/dataset.yaml ->  local_path: \"{path.parent.as_posix()}\"")


if __name__ == "__main__":
    main()
