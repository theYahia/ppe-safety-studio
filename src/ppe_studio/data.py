"""Dataset acquisition.

Primary path: download an open PPE dataset from Roboflow Universe (Hard Hat Workers — CC0,
or PPE v2). Fallback: point at an already-extracted YOLO dataset on disk. Either way the
function returns the path to the Ultralytics `data.yaml`.

No dataset slug/version is hardcoded in code — they live in configs/dataset.yaml so the repo
stays honest about what it pulls. Set ROBOFLOW_API_KEY in the environment (or .env.local).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def _read_env_local(start: Path) -> None:
    """Load KEY=VALUE pairs from the nearest .env.local walking up from `start`."""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        env = parent / ".env.local"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


def resolve_dataset(config_path: str | Path = "configs/dataset.yaml") -> Path:
    """Return the path to a ready-to-train data.yaml.

    If `local_path` in the config exists, use it directly. Otherwise download the
    configured Roboflow dataset into `download_dir`.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    local = cfg.get("local_path")
    if local:
        local_yaml = Path(local)
        if local_yaml.is_dir():
            local_yaml = local_yaml / "data.yaml"
        if local_yaml.exists():
            return local_yaml

    rf_cfg = cfg["roboflow"]
    _read_env_local(Path.cwd())
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not set and no valid local_path in dataset config. "
            "Add it to .env.local or set local_path to an extracted YOLO dataset."
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(rf_cfg["workspace"]).project(rf_cfg["project"])
    dataset = project.version(int(rf_cfg["version"])).download(
        rf_cfg.get("format", "yolov11"),
        location=cfg.get("download_dir", "data/ppe"),
    )
    return Path(dataset.location) / "data.yaml"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Resolve / download the PPE dataset.")
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    print(resolve_dataset(args.config))
