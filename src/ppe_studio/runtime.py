"""Runtime helpers: GPU selection and class→role mapping.

The workstation has two CUDA GPUs (GTX 1070 Ti 8GB + RTX 3080 10GB). `resolve_device`
defaults to the device with the most total memory so training lands on the 3080 without
relying on CUDA's bus ordering.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def resolve_device(requested: str = "auto") -> str | int:
    """Return an Ultralytics-compatible device.

    "auto" → CUDA index with the largest total memory, else CPU.
    Any explicit value ("0", "1", "cpu") passes through unchanged.
    """
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    best_idx, best_mem = 0, -1
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory
        if mem > best_mem:
            best_idx, best_mem = i, mem
    return best_idx


def load_class_roles(config_path: str | Path) -> dict[str, str]:
    """Map dataset class names → semantic roles for the geofencing layer.

    Roles: "subject" (person), "ppe_ok" (helmet/vest present),
    "ppe_violation" (bare head / missing PPE). Defined in configs/classes.yaml.
    """
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data["roles"].items()}


def load_zones(config_path: str | Path):
    """Load danger zones from configs/zones.yaml into Zone objects."""
    from .geofencing import zone_from_config

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return [zone_from_config(z) for z in raw.get("zones", [])]
