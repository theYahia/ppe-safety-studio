"""Inference pipeline: YOLO detections → normalized Detection objects → violations.

Keeps the model layer thin. Geofencing/compliance lives in geofencing.py so it can be
tested and reused without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .geofencing import Detection, Violation, Zone, evaluate
from .runtime import load_class_roles, resolve_device


@dataclass
class FrameResult:
    detections: list[Detection]
    violations: list[Violation]
    width: int
    height: int


class PPEDetector:
    """Thin wrapper around an Ultralytics YOLO model + the compliance layer."""

    def __init__(
        self,
        weights: str | Path,
        classes_config: str | Path,
        zones: list[Zone] | None = None,
        device: str = "auto",
        conf: float = 0.35,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.roles = load_class_roles(classes_config)
        self.zones = zones or []
        self.device = resolve_device(device)
        self.conf = conf

    def _to_detections(self, boxes, names, w: int, h: int) -> list[Detection]:
        out: list[Detection] = []
        for b in boxes:
            cls_name = names[int(b.cls)]
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            track_id = int(b.id) if getattr(b, "id", None) is not None else None
            out.append(
                Detection(
                    cls_name=cls_name,
                    role=self.roles.get(cls_name, "subject"),
                    conf=float(b.conf),
                    xyxy=(x1 / w, y1 / h, x2 / w, y2 / h),
                    track_id=track_id,
                )
            )
        return out

    def predict_image(self, image) -> FrameResult:
        """Run detection on a single image (path, PIL, or ndarray)."""
        results = self.model.predict(image, device=self.device, conf=self.conf, verbose=False)
        r = results[0]
        h, w = r.orig_shape
        dets = self._to_detections(r.boxes, r.names, w, h)
        return FrameResult(dets, evaluate(dets, self.zones), w, h)

    def track_video(self, source) -> Iterator[FrameResult]:
        """Stream detections with ByteTrack IDs — yields one FrameResult per frame."""
        stream = self.model.track(
            source=source,
            device=self.device,
            conf=self.conf,
            tracker="bytetrack.yaml",
            stream=True,
            persist=True,
            verbose=False,
        )
        for r in stream:
            h, w = r.orig_shape
            dets = self._to_detections(r.boxes, r.names, w, h)
            yield FrameResult(dets, evaluate(dets, self.zones), w, h)
