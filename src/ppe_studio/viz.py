"""Drawing helpers — render detections, zones, and violations onto an image.

Palette follows impeccable.style: tinted neutrals, no pure red/green, no gradients.
Used by both the Streamlit demo and README screenshot generation.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# Tinted, calm palette (OKLCH-ish, mapped to RGB). No garish primaries.
COLORS = {
    "ppe_ok": (96, 158, 120),        # muted green
    "ppe_violation": (200, 96, 88),  # muted clay-red
    "subject": (120, 132, 156),      # slate
    "zone_warning": (210, 168, 96),  # ochre
    "zone_critical": (196, 92, 84),  # clay
    "text": (38, 40, 46),
    "panel": (244, 242, 238),
}


def _font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render(image: Image.Image, frame, zones) -> Image.Image:
    """Return a copy of `image` with zones, boxes, and a violation panel drawn."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    font = _font(16)
    small = _font(13)

    # Danger zones (filled translucent + outline).
    for z in zones:
        pts = [(x * w, y * h) for x, y in z.polygon]
        fill = COLORS["zone_critical"] if z.severity.value == "critical" else COLORS["zone_warning"]
        draw.polygon(pts, fill=(*fill, 48), outline=(*fill, 220), width=3)
        draw.text((pts[0][0] + 4, pts[0][1] + 4), z.name, fill=COLORS["text"], font=small)

    # Detections.
    for d in frame.detections:
        x1, y1, x2, y2 = d.xyxy[0] * w, d.xyxy[1] * h, d.xyxy[2] * w, d.xyxy[3] * h
        color = COLORS.get(d.role, COLORS["subject"])
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=3)
        tag = d.cls_name if d.track_id is None else f"{d.cls_name}#{d.track_id}"
        draw.text((x1, max(0, y1 - 16)), f"{tag} {d.conf:.2f}", fill=color, font=small)

    # Violation panel (top-left), each line tinted by severity.
    if frame.violations:
        pad = 8
        box_h = 22 * len(frame.violations) + 2 * pad
        draw.rectangle([0, 0, w, box_h], fill=(*COLORS["panel"], 235))
        sev_color = {
            "critical": COLORS["zone_critical"],
            "warning": COLORS["zone_warning"],
            "info": COLORS["subject"],
        }
        for i, v in enumerate(frame.violations):
            col = sev_color.get(v.severity.value, COLORS["text"])
            draw.text(
                (pad, pad + i * 22),
                f"{v.severity.value.upper()}: {v.message}",
                fill=col,
                font=font,
            )
    return img
