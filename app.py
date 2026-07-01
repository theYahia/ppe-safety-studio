"""PPE Safety Studio — Streamlit demo (Hugging Face Space entry point).

Upload a frame → detect PPE → check operator-defined danger zones → list violations.
Open-source learning artifact, not a commercial safety system. See README for limits.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from ppe_studio.runtime import load_zones  # noqa: E402

st.set_page_config(page_title="PPE Safety Studio", page_icon="🦺", layout="wide")

# impeccable.style: tinted neutrals, generous spacing, no gradient text / glass.
st.markdown(
    """
    <style>
      :root { --ink:#26282e; --muted:#6b6f78; --bg:#f7f5f1; }
      .stApp { background: var(--bg); }
      h1,h2,h3 { color: var(--ink); letter-spacing:-0.01em; }
      .block-container { max-width: 1100px; padding-top: 2rem; }
      p, label, .stMarkdown { color: var(--ink); line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PPE Safety Studio")
st.markdown(
    "Детекция СИЗ (каска) и контроль опасных зон на одном кадре. "
    "YOLOv11 + polygon-geofencing. Открытый учебный проект — не сертифицированная система ТБ."
)

with st.sidebar:
    st.header("Параметры")
    weights = st.text_input("Веса модели", value="runs/ppe_yolo11s/weights/best.pt")
    conf = st.slider("Порог уверенности", 0.1, 0.9, 0.35, 0.05)
    use_zones = st.checkbox("Включить опасные зоны", value=True)
    st.caption("Зоны редактируются в `configs/zones.yaml`. Роли классов — `configs/classes.yaml`.")

uploaded = st.file_uploader("Загрузите изображение со стройки/цеха", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Загрузите кадр, чтобы увидеть детекции, зоны и список нарушений.")
    st.stop()

image = Image.open(uploaded)

if not Path(weights).exists():
    st.warning(
        f"Веса не найдены: `{weights}`. Обучите модель "
        "(`python -m ppe_studio.train`) или укажите путь к `best.pt`."
    )
    st.image(image, caption="Исходный кадр", use_container_width=True)
    st.stop()

from ppe_studio.infer import PPEDetector  # noqa: E402  (import after weights check — avoids slow load on error)
from ppe_studio.viz import render  # noqa: E402

zones = load_zones(ROOT / "configs" / "zones.yaml") if use_zones else []

@st.cache_resource
def _load(weights_path: str, conf_v: float):
    return PPEDetector(weights_path, ROOT / "configs" / "classes.yaml", conf=conf_v)

detector = _load(weights, conf)
detector.zones = zones

with st.spinner("Инференс…"):
    frame = detector.predict_image(image)
    rendered = render(image, frame, zones)

col1, col2 = st.columns([3, 2])
with col1:
    st.image(rendered, caption="Детекции + зоны", use_container_width=True)
with col2:
    st.subheader("Нарушения")
    if not frame.violations:
        st.success("Нарушений не обнаружено.")
    else:
        for v in frame.violations:
            icon = {"critical": "🟥", "warning": "🟧", "info": "🟦"}[v.severity.value]
            st.markdown(f"{icon} **{v.severity.value.upper()}** — {v.message}")
    st.divider()
    st.caption(
        f"Детекций: {len(frame.detections)} · "
        f"людей: {sum(1 for d in frame.detections if d.role == 'subject')} · "
        f"без каски: {sum(1 for d in frame.detections if d.role == 'ppe_violation')}"
    )
