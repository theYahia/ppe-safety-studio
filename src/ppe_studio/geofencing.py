"""Danger-zone geofencing and PPE-compliance logic.

Pure geometry + rules — no GPU, no model dependency, fully unit-testable.
This is the differentiator that lifts the project out of "yet another YOLO detector":
detections are interpreted against operator-defined danger zones and PPE requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test (no external deps).

    Coordinates can be pixel or normalized — both endpoints must share a system.
    A point exactly on an edge is treated deterministically (may count as inside);
    that ambiguity is irrelevant at detection resolution.
    """
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Zone:
    """An operator-defined polygon with PPE requirements.

    polygon: normalized [0,1] coordinates so a zone survives frame resize.
    requires: PPE roles that must be present for a subject inside the zone.
    severity: how serious a violation inside this zone is.
    """
    name: str
    polygon: tuple[Point, ...]
    requires: frozenset[str] = field(default_factory=frozenset)
    severity: Severity = Severity.WARNING


@dataclass(frozen=True)
class Detection:
    """One model detection, normalized to [0,1] image coordinates."""
    cls_name: str
    role: str  # mapped role: "subject" | "ppe_ok" | "ppe_violation"
    conf: float
    xyxy: tuple[float, float, float, float]  # x1,y1,x2,y2 normalized
    track_id: int | None = None

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot(self) -> Point:
        """Bottom-center — the contact point used to decide zone membership."""
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)


@dataclass(frozen=True)
class Violation:
    rule: str  # e.g. "no_helmet", "intrusion"
    severity: Severity
    detection: Detection
    zone: Zone | None
    message: str


def _iou_x(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Horizontal overlap ratio — cheap proxy to associate a bare head with a person."""
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    union = max(ax2, bx2) - min(ax1, bx1)
    return inter / union if union > 0 else 0.0


def evaluate(
    detections: Iterable[Detection],
    zones: Sequence[Zone],
    site_requires: frozenset[str] = frozenset({"hardhat"}),
) -> list[Violation]:
    """Turn raw detections into PPE/zone violations.

    Two rule families:
      * no_helmet — a `ppe_violation` role (e.g. a bare "head") is a PPE breach.
        Severity escalates to the zone's level when the breach sits inside a zone.
      * intrusion — a `subject` (person) inside a zone is flagged at the zone severity
        when the zone forbids unprotected presence (modeled via `requires`).

    `site_requires` is the baseline PPE expected everywhere on site.
    """
    dets = list(detections)
    violations: list[Violation] = []

    for det in dets:
        if det.role != "ppe_violation":
            continue
        hit_zone = next((z for z in zones if point_in_polygon(det.foot, z.polygon)), None)
        if hit_zone is not None:
            sev = (
                Severity.CRITICAL
                if hit_zone.severity == Severity.CRITICAL
                else Severity.WARNING
            )
            where = f"в зоне «{hit_zone.name}»"
            zone = hit_zone
        elif site_requires:
            sev = Severity.WARNING
            where = "на площадке"
            zone = None
        else:
            continue
        violations.append(
            Violation(
                rule="no_helmet",
                severity=sev,
                detection=det,
                zone=zone,
                message=f"Нет каски {where} (conf {det.conf:.2f})",
            )
        )

    # Person present inside a zone that requires PPE but no helmet detected nearby.
    persons = [d for d in dets if d.role == "subject"]
    helmets = [d for d in dets if d.role == "ppe_ok"]
    for person in persons:
        zone = next((z for z in zones if point_in_polygon(person.foot, z.polygon)), None)
        if zone is None or not zone.requires:
            continue
        covered = any(_iou_x(person.xyxy, h.xyxy) > 0.3 for h in helmets)
        if not covered:
            violations.append(
                Violation(
                    rule="intrusion",
                    severity=zone.severity,
                    detection=person,
                    zone=zone,
                    message=f"Человек без подтверждённой каски в зоне «{zone.name}»",
                )
            )
    return violations


def zone_from_config(raw: dict) -> Zone:
    """Build a Zone from a configs/zones.yaml entry."""
    return Zone(
        name=raw["name"],
        polygon=tuple((float(x), float(y)) for x, y in raw["polygon"]),
        requires=frozenset(raw.get("requires", [])),
        severity=Severity(raw.get("severity", "warning")),
    )
