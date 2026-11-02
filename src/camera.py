"""
Camera backends for the live inspection station.

Orbbec Femto Bolt exposes several UVC devices (RGB, IR, depth). Opening the
wrong one, or forcing 1280x720 onto an IR/depth endpoint, produces gray
horizontal static. We open at native size, score colorfulness, and prefer
the RGB stream.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import CAMERA_PROBE_MAX_INDEX

BACKENDS = []
if hasattr(cv2, "CAP_DSHOW"):
    BACKENDS.append(("DirectShow", cv2.CAP_DSHOW))
if hasattr(cv2, "CAP_MSMF"):
    BACKENDS.append(("MSMF", cv2.CAP_MSMF))
BACKENDS.append(("Any", cv2.CAP_ANY))


def _fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def frame_quality(bgr: Optional[np.ndarray]) -> dict:
    """Decide whether a grabbed frame is usable RGB, or IR/depth/static."""
    info = {
        "ok": False,
        "reason": "empty",
        "width": 0,
        "height": 0,
        "channels": 0,
        "colorfulness": 0.0,
        "saturation": 0.0,
        "std": 0.0,
    }
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return info
    if bgr.ndim == 2:
        h, w = bgr.shape
        info.update(width=w, height=h, channels=1, reason="grayscale_ir_or_depth")
        return info
    h, w = bgr.shape[:2]
    c = bgr.shape[2] if bgr.ndim == 3 else 1
    info.update(width=w, height=h, channels=c)
    if c != 3 or min(h, w) < 240:
        info["reason"] = "unsupported_layout"
        return info

    b, g, r = cv2.split(bgr.astype(np.float32))
    colorfulness = float((np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))) / 2.0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))
    std = float(np.std(bgr))
    row_hf = float(np.mean(np.abs(np.diff(g, axis=1))))
    info.update(colorfulness=colorfulness, saturation=saturation, std=std)

    # Identical R/G/B → IR duplicated into 3 channels.
    if colorfulness < 1.5:
        info["reason"] = "ir_or_depth_not_rgb"
        return info
    # YUY2/depth mis-decoded as BGR: gray stripes, high horizontal frequency.
    if saturation < 12 and row_hf > 18:
        info["reason"] = "undecoded_or_wrong_endpoint"
        return info
    if std < 8:
        info["reason"] = "flat_or_no_signal"
        return info
    info["ok"] = True
    info["reason"] = "ok"
    return info


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _try_open(
    index: int,
    backend_id: int,
    fourcc: Optional[str] = None,
    set_size: bool = False,
) -> Tuple[Optional[cv2.VideoCapture], Optional[np.ndarray], dict]:
    cap = cv2.VideoCapture(int(index), backend_id)
    if not cap.isOpened():
        return None, None, {"ok": False, "reason": "not_opened"}
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, _fourcc(fourcc))
    if set_size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    ok = False
    for _ in range(4):
        ok, frame = cap.read()
    if frame is None:
        cap.release()
        return None, None, {"ok": False, "reason": "no_frame"}
    quality = frame_quality(frame)
    return cap, frame, quality


def open_color_capture(index: int) -> Tuple[Optional[cv2.VideoCapture], dict]:
    """
    Open one device index. Native resolution first — forcing 1280x720 onto a
    Femto IR/depth endpoint produces gray static.
    """
    attempts = []
    order = []
    for name, backend_id in BACKENDS:
        order.append((name, backend_id, "MJPG", False))
        order.append((name, backend_id, None, False))
    for name, backend_id in BACKENDS[:2]:
        order.append((name, backend_id, "MJPG", True))

    for name, backend_id, fourcc, set_size in order:
        cap, frame, quality = _try_open(index, backend_id, fourcc, set_size)
        label = f"{name}/{fourcc or 'default'}/{'1280x720' if set_size else 'native'}"
        attempts.append((label, quality.get("reason")))
        if cap is None:
            continue
        if quality.get("ok"):
            meta = {
                **quality,
                "backend": name,
                "fourcc": fourcc or "default",
                "index": index,
                "attempts": attempts,
            }
            return cap, meta
        cap.release()

    cap, frame, quality = _try_open(index, BACKENDS[0][1], None, False)
    meta = {
        **(quality or {"ok": False, "reason": "no_frame"}),
        "backend": BACKENDS[0][0],
        "fourcc": "default",
        "index": index,
        "attempts": attempts,
    }
    return cap, meta


def list_opencv_cameras(max_index: int = CAMERA_PROBE_MAX_INDEX) -> List[dict]:
    found = []
    for i in range(max_index):
        cap, meta = open_color_capture(i)
        if cap is None:
            continue
        cap.release()
        found.append(
            {
                "index": i,
                "ok_rgb": bool(meta.get("ok")),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "reason": meta.get("reason"),
                "colorfulness": round(float(meta.get("colorfulness") or 0), 1),
                "backend": meta.get("backend"),
            }
        )
    return found


def pick_best_rgb_index(preferred: int, scanned: List[dict]) -> Optional[int]:
    rgb = [d for d in scanned if d.get("ok_rgb")]
    if not rgb:
        return None
    for d in rgb:
        if d["index"] == preferred:
            return preferred
    rgb.sort(key=lambda d: d.get("colorfulness", 0), reverse=True)
    return int(rgb[0]["index"])


def _color_frame_to_bgr(color_frame) -> Optional[np.ndarray]:
    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    raw = np.asanyarray(color_frame.get_data())
    fmt = str(color_frame.get_format())
    if raw.size == 0:
        return None
    if "MJPG" in fmt or "MJPEG" in fmt:
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if raw.size == width * height * 3:
        img = raw.reshape((height, width, 3))
        if "RGB" in fmt:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if raw.size == width * height * 2:
        img = raw.reshape((height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUY2)
    return None


class StationCamera:
    """Single RGB grabber: Orbbec pipeline or OpenCV color device."""

    def __init__(self):
        self.backend = "none"
        self.detail = ""
        self.index = 0
        self.quality = {}
        self._cap = None
        self._pipeline = None

    def open(self, index: int = 0, prefer_orbbec: bool = True) -> Tuple[bool, str]:
        self.close()
        self.index = int(index)
        if prefer_orbbec:
            ok, msg = self._open_orbbec()
            if ok:
                return True, msg
        return self._open_opencv(self.index)

    def _open_orbbec(self) -> Tuple[bool, str]:
        try:
            from pyorbbecsdk import Config, OBSensorType, Pipeline  # type: ignore
        except Exception:
            try:
                from pyorbbecsdk2 import Config, OBSensorType, Pipeline  # type: ignore
            except Exception:
                return False, "Orbbec SDK Python package is not installed; using OpenCV."
        try:
            pipeline = Pipeline()
            config = Config()
            profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profiles.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            pipeline.start(config)
            frames = pipeline.wait_for_frames(2000)
            if frames is None or frames.get_color_frame() is None:
                pipeline.stop()
                return False, "Orbbec device opened but no RGB frame yet."
            self._pipeline = pipeline
            self.backend = "orbbec"
            self.detail = "Orbbec Femto RGB (SDK color sensor)"
            self.quality = {"ok": True, "reason": "orbbec_sdk"}
            return True, self.detail
        except Exception as exc:  # noqa: BLE001
            self._pipeline = None
            return False, f"Orbbec SDK present but camera failed ({exc}). Trying OpenCV."

    def _open_opencv(self, index: int) -> Tuple[bool, str]:
        cap, meta = open_color_capture(index)
        if cap is None:
            return False, (
                f"Could not open camera index {index}. "
                "Close Orbbec Viewer, then Scan cameras."
            )
        self._cap = cap
        self.backend = "opencv"
        self.quality = meta
        w, h = meta.get("width"), meta.get("height")
        if meta.get("ok"):
            self.detail = (
                f"RGB camera index {index} · {w}x{h} · {meta.get('backend')} "
                f"({meta.get('fourcc')})"
            )
            return True, self.detail
        self.detail = (
            f"Camera index {index} opened ({w}x{h}) but this is not an RGB picture "
            f"({meta.get('reason')}). Femto IR/depth looks like gray static — "
            f"raise the camera index or click Scan cameras and pick an RGB device."
        )
        return True, self.detail

    def read_rgb(self, flush: bool = False) -> Optional[np.ndarray]:
        if self.backend == "orbbec" and self._pipeline is not None:
            frames = self._pipeline.wait_for_frames(30)
            if frames is None:
                return None
            color = frames.get_color_frame()
            if color is None:
                return None
            bgr = _color_frame_to_bgr(color)
            if bgr is None:
                return None
            return bgr_to_rgb(bgr)
        if self._cap is not None:
            last = None
            # DirectShow/MSMF often returns an empty grab after a pause.
            # Do not call grab() — it stalls this camera. Read a few times instead.
            tries = 4 if flush else 2
            for _ in range(tries):
                _ok, bgr = self._cap.read()
                if bgr is not None and getattr(bgr, "size", 0) > 0:
                    last = bgr
            if last is None:
                return None
            return bgr_to_rgb(last)
        return None

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self.backend = "none"

    def __del__(self):
        self.close()
