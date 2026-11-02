"""Industrial live inspection: GOOD or DEFECTIVE. Care Yu branded station."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.camera import StationCamera, list_opencv_cameras
from src.config import CAMERA_DEFAULT_INDEX, LOGO_MARK_PATH, LOGO_PATH
from src.inspection import inspect_image
from src.presence import PresenceTrigger

st.set_page_config(
    page_title="Care Yu · Fitting Inspection",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; height: 0; }
.stDeployButton { display: none; }
.block-container { padding-top: 0.6rem; padding-bottom: 2rem; max-width: 1280px; }
.stApp {
  background:
    radial-gradient(1100px 520px at 0% 0%, rgba(120, 186, 255, 0.45), transparent 58%),
    radial-gradient(900px 480px at 100% 10%, rgba(186, 220, 255, 0.55), transparent 52%),
    linear-gradient(180deg, #ffffff 0%, #eaf4ff 42%, #f7fbff 100%);
  color: #1b2a44;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f4f9ff 0%, #e7f2ff 100%);
  border-right: 1px solid #c5ddf7;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

.cy-header {
  display: flex; align-items: center; gap: 18px;
  padding: 10px 16px 16px 4px;
  margin-bottom: 8px;
  border-bottom: 1px solid #c5ddf7;
}
.cy-logo {
  height: 72px; width: auto; display: block;
  background: transparent; padding: 0;
  filter: drop-shadow(0 3px 8px rgba(16, 35, 63, 0.14));
}
.cy-logo-side {
  width: 168px; height: auto; display: block; margin: 0 auto 14px auto;
  background: transparent; padding: 0;
  filter: drop-shadow(0 3px 8px rgba(16, 35, 63, 0.12));
}
.cy-titles { display: flex; flex-direction: column; gap: 2px; }
.cy-brand {
  font-size: 13px; font-weight: 700; letter-spacing: 0.22em;
  color: #3a5f8c; margin: 0;
}
.cy-brand span { color: #1f63c7; }
.cy-title {
  font-size: 28px; font-weight: 800; letter-spacing: 0.06em;
  color: #10233f; margin: 0; line-height: 1.1;
}
.cy-sub { margin: 4px 0 0 0; color: #4e6788; font-size: 14px; }

.cy-station {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(380px, 0.9fr);
  gap: 20px;
  align-items: stretch;
  width: 100%;
}
.cy-feed img {
  width: 100%; height: auto; display: block;
  border-radius: 12px; border: 1px solid #c5ddf7;
  box-shadow: 0 10px 28px rgba(47, 125, 225, 0.12);
  background: #fff;
}
.cy-hint {
  margin: 10px 0 0 0; color: #4e6788; font-size: 13px;
}
.cy-aside {
  display: flex; flex-direction: column; justify-content: center;
  min-height: 280px;
}

.iv-status {
  border-radius: 14px; padding: 36px 16px; text-align: center;
  margin: 0;
  min-height: 220px;
  display: flex; flex-direction: column; justify-content: center;
  overflow: hidden;
}
.iv-status .lbl {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px; letter-spacing: 0.28em; margin-bottom: 14px; opacity: 0.85;
}
.iv-status .val {
  font-size: clamp(28px, 3.4vw, 40px);
  font-weight: 800; letter-spacing: 0.04em; line-height: 1.15;
  white-space: nowrap;
}
.iv-pass {
  background: linear-gradient(180deg, #e9fff3 0%, #d4f8e4 100%);
  border: 1px solid #2ecc71; color: #0b7a3f;
  box-shadow: 0 8px 24px rgba(46, 204, 113, 0.16);
}
.iv-fail {
  background: linear-gradient(180deg, #ffecec 0%, #ffd6d6 100%);
  border: 1px solid #e74c3c; color: #b42318;
  box-shadow: 0 8px 24px rgba(231, 76, 60, 0.16);
}
.iv-wait {
  background: linear-gradient(180deg, #ffffff 0%, #e7f3ff 100%);
  border: 1px solid #b7d4f5; color: #3a5f8c;
}
.iv-note {
  font-size: 13px; color: #3a5f8c;
  border: 1px solid #c5ddf7; background: #ffffff;
  padding: 12px 14px; border-radius: 10px;
}
.cy-side-cap {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; letter-spacing: 0.22em; color: #5a7aa0; margin: 0 0 10px 0;
}
div.stButton > button[kind="primary"],
div.stButton > button {
  border-radius: 10px; font-weight: 700; letter-spacing: 0.08em;
}
div.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #4aa3ff 0%, #1f63c7 100%);
  border: 1px solid #7ec0ff; color: #fff;
}
[data-testid="stImage"] img {
  border-radius: 12px; border: 1px solid #c5ddf7;
  box-shadow: 0 10px 28px rgba(47, 125, 225, 0.12);
}
div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
</style>
"""


def _img_data_uri(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def brand_header() -> None:
    uri = _img_data_uri(LOGO_MARK_PATH if LOGO_MARK_PATH.exists() else LOGO_PATH)
    img = f'<img class="cy-logo" src="{uri}" alt="Care Yu Automation" />' if uri else ""
    st.markdown(
        f"""
        <div class="cy-header">
          {img}
          <div class="cy-titles">
            <h1 class="cy-title">FITTING INSPECTION</h1>
            <p class="cy-sub">Station 01 · Place the part. Result is GOOD or DEFECTIVE.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_panel(label: str) -> str:
    cls = {"GOOD": "iv-pass", "DEFECTIVE": "iv-fail"}.get(label, "iv-wait")
    return (
        f'<div class="iv-status {cls}">'
        f'<div class="lbl">RESULT</div>'
        f'<div class="val">{label}</div>'
        f"</div>"
    )


def _b64_jpeg(rgb, max_side=720) -> str:
    import cv2

    im = _preview(rgb, max_side)
    bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_station(rgb, caption: str, result: dict | None, waiting_note: str) -> None:
    """Camera on the left, result on the right — one HTML grid, no wrap."""
    img = ""
    if rgb is not None:
        b64 = _b64_jpeg(rgb)
        if b64:
            img = f'<img src="data:image/jpeg;base64,{b64}" alt="Live RGB" />'
    verdict = "WAITING" if result is None else result["verdict"]
    note = waiting_note if result is None else ""
    aside_note = f'<p class="cy-hint">{note}</p>' if note else ""
    st.markdown(
        f"""
        <div class="cy-station">
          <div class="cy-feed">
            {img}
            <p class="cy-hint">{caption}</p>
          </div>
          <div class="cy-aside">
            {status_panel(verdict)}
            {aside_note}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _preview(im, max_side=520):
    if im is None:
        return im
    if getattr(im, "size", 1) == 0:
        return im
    h, w = im.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return im
    s = max_side / float(m)
    import cv2

    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _too_blurry(rgb) -> bool:
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var()) < 16.0


def analyze_rgb(rgb: np.ndarray, live: bool = True) -> dict | None:
    """GOOD unless a real defect is found. Clean cream fittings stay GOOD."""
    result = inspect_image(
        rgb,
        params={"live_mode": live, "skip_live_save": not live},
        filename="live.png" if live else "frame.png",
    )
    if result.get("no_part"):
        return None
    return {
        "verdict": "GOOD" if result.get("status") == "PASS" else "DEFECTIVE",
        "image": rgb,
    }


def render_result(result: dict | None, waiting_note: str) -> None:
    if result is None:
        st.markdown(status_panel("WAITING"), unsafe_allow_html=True)
        st.caption(waiting_note)
        return
    st.markdown(status_panel(result["verdict"]), unsafe_allow_html=True)


def _close_camera():
    if "station_cam" in st.session_state:
        try:
            st.session_state.station_cam.close()
        except Exception:
            pass
        st.session_state.pop("station_cam", None)
        st.session_state.pop("cam_msg", None)
        st.session_state.pop("live_frame", None)
        st.session_state.pop("presence", None)
        st.session_state.pop("cam_miss", None)


st.markdown(CSS, unsafe_allow_html=True)
brand_header()

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = None

with st.sidebar:
    side_uri = _img_data_uri(LOGO_PATH)
    if side_uri:
        st.markdown(
            f'<img class="cy-logo-side" src="{side_uri}" alt="Care Yu Automation" />',
            unsafe_allow_html=True,
        )
    st.markdown('<p class="cy-side-cap">CAMERA SETUP</p>', unsafe_allow_html=True)
    cam_idx = st.number_input("Camera index", 0, 8, CAMERA_DEFAULT_INDEX)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Scan cameras", width="stretch"):
            st.session_state.cam_scan = list_opencv_cameras()
    with b2:
        if st.button("Reconnect", width="stretch"):
            _close_camera()
            st.rerun()
    for c in st.session_state.get("cam_scan") or []:
        st.caption(
            f"index {c.get('index')} · {c.get('width')}x{c.get('height')} · {c.get('reason')}"
        )

src = st.radio("Input", ["Live camera", "Upload image"], horizontal=True, index=0)


@st.fragment(run_every=0.35)
def _live_preview_loop():
    cam = st.session_state.get("station_cam")
    if cam is None:
        render_station(None, "Camera not open.", None, "Open the camera to inspect.")
        return
    frame = cam.read_rgb(flush=False)
    if frame is None:
        miss = int(st.session_state.get("cam_miss") or 0) + 1
        st.session_state.cam_miss = miss
        held = st.session_state.get("live_frame")
        render_station(
            held,
            "If this stays blank, click Reconnect in the sidebar.",
            st.session_state.analysis,
            "Waiting for a live frame.",
        )
        if miss == 12:
            idx = int(getattr(cam, "index", CAMERA_DEFAULT_INDEX))
            try:
                cam.close()
            except Exception:
                pass
            cam = StationCamera()
            _ok, msg = cam.open(idx, prefer_orbbec=False)
            st.session_state.station_cam = cam
            st.session_state.cam_msg = msg
            st.session_state.cam_miss = 0
        return
    st.session_state.cam_miss = 0
    st.session_state.live_frame = frame
    disp = _preview(frame, 720)
    if "presence" not in st.session_state:
        st.session_state.presence = PresenceTrigger(stable_needed=3, empty_needed=3)
    trig = st.session_state.presence.update(disp)
    if trig.phase == "EMPTY":
        st.session_state.analysis = None
    skipped_blur = False
    if trig.fire:
        if _too_blurry(frame):
            st.session_state.presence.mark_inspect_failed()
            skipped_blur = True
        else:
            try:
                st.session_state.analysis = analyze_rgb(frame)
            except Exception as exc:
                st.session_state.last_error = str(exc)
            st.session_state.presence.remember_inspect_frame(frame)
    if trig.phase == "EMPTY":
        cap = "No part. Place the fitting in the camera view."
        shown = None
    elif trig.phase == "WAIT" or skipped_blur:
        cap = "Part seen. Hold still — inspecting automatically."
        shown = None
    elif (st.session_state.analysis or {}).get("verdict") == "GOOD":
        cap = "GOOD. Remove or swap for the next part."
        shown = st.session_state.analysis
    elif (st.session_state.analysis or {}).get("verdict") == "DEFECTIVE":
        cap = "DEFECTIVE. Remove or swap for the next part."
        shown = st.session_state.analysis
    else:
        cap = "Hold still, or click INSPECT THIS FRAME."
        shown = st.session_state.analysis
    render_station(
        disp,
        cap,
        shown,
        "Place a fitting in the live view, or click INSPECT THIS FRAME.",
    )
    if st.button("INSPECT THIS FRAME", type="primary", width="stretch"):
        grabbed = st.session_state.get("live_frame")
        if grabbed is not None:
            st.session_state.analysis = analyze_rgb(grabbed)
            if "presence" in st.session_state:
                st.session_state.presence.remember_inspect_frame(grabbed)


if src == "Live camera":
    if "live_cam_on" not in st.session_state:
        st.session_state.live_cam_on = True
    live_on = st.toggle("Open camera", key="live_cam_on")
    st.caption("Put the part in the picture. Hold still. The result appears by itself.")
    if live_on:
        need_open = "station_cam" not in st.session_state
        cam_obj = st.session_state.get("station_cam")
        if cam_obj is not None and int(getattr(cam_obj, "index", -1)) != int(cam_idx):
            _close_camera()
            need_open = True
        if need_open:
            cam = StationCamera()
            ok, msg = cam.open(int(cam_idx), prefer_orbbec=False)
            st.session_state.station_cam = cam
            st.session_state.cam_msg = msg
            st.session_state.presence = PresenceTrigger(stable_needed=3, empty_needed=3)
            st.session_state.cam_miss = 0
            warm = cam.read_rgb(flush=True)
            if warm is not None:
                st.session_state.live_frame = warm
        st.caption(st.session_state.get("cam_msg") or "")
        _live_preview_loop()
    else:
        _close_camera()
        st.session_state.analysis = None
        st.markdown(
            '<div class="iv-note">Turn on Open camera to start live inspection.</div>',
            unsafe_allow_html=True,
        )
else:
    _close_camera()
    shot, panel = st.columns([1.35, 0.75], gap="large")
    with shot:
        uploaded = st.file_uploader(
            "Upload NEW image",
            type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
            accept_multiple_files=False,
        )
        query_image = None
        upload_key = None
        if uploaded is not None:
            upload_key = (uploaded.name, int(uploaded.size))
            try:
                query_image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
            except Exception:
                st.error("Could not read that file as an image.")
                query_image = None

        if upload_key != st.session_state.upload_key:
            st.session_state.upload_key = upload_key
            st.session_state.analysis = None

        analyze = st.button("ANALYZE", type="primary", width="stretch", disabled=query_image is None)

        if query_image is not None:
            st.image(query_image, caption="Selected image", width="stretch")
        else:
            st.markdown(
                '<div class="iv-note">No image selected. Upload a photo of the fitting.</div>',
                unsafe_allow_html=True,
            )

        if query_image is not None and analyze:
            st.session_state.analysis = analyze_rgb(np.array(query_image), live=False)

    with panel:
        render_result(
            st.session_state.analysis,
            "Result appears after ANALYZE. The filename is not used.",
        )
