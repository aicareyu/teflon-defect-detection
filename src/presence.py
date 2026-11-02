"""
Auto-trigger: inspect once when a fitting is placed and still.

EMPTY → WAIT (part seen) → FIRE (stable) → HOLD until the part is removed.
Presence uses the white plastic body so a wooden table is not treated as a part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.config import (
    LIVE_EMPTY_AREA_MAX,
    LIVE_EMPTY_FRAMES,
    LIVE_MIN_BBOX_HEIGHT_FRAC,
    LIVE_MIN_BBOX_WIDTH_FRAC,
    LIVE_MOTION_STABLE,
    LIVE_PRESENT_AREA_MIN,
    LIVE_STABLE_FRAMES,
)


def detect_product_blob(rgb: np.ndarray) -> dict:
    """Cheap live occupancy: smooth cream plastic plus nearby brass. Wood grain is not a part."""
    out = {
        "found": False,
        "area_ratio": 0.0,
        "bbox": None,
        "reason": "empty_image",
        "white_ratio": 0.0,
    }
    if rgb is None or rgb.size == 0:
        return out
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (5, 5), 0)
    # Cream PP is bright, desaturated, and smooth. Wood highlights are textured.
    plastic = (
        (L > 168)
        & (np.abs(A.astype(np.int16) - 128) < 12)
        & (B > 118)
        & (B < 150)
        & (S < 42)
        & (V > 160)
        & (tex < 14.0)
    )
    white = np.uint8(plastic) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, tiny)
    brass = cv2.inRange(hsv, (10, 70, 80), (38, 255, 255))
    brass_near = cv2.bitwise_and(brass, cv2.dilate(white, kernel, iterations=2))
    product = cv2.bitwise_or(white, brass_near)
    product = cv2.morphologyEx(product, cv2.MORPH_CLOSE, kernel)
    white_ratio = float(cv2.countNonZero(white)) / float(small.shape[0] * small.shape[1])
    out["white_ratio"] = white_ratio
    contours, _ = cv2.findContours(product, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        out["reason"] = "no_contour"
        return out
    contour = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(contour)) / float(small.shape[0] * small.shape[1])
    x, y, bw, bh = cv2.boundingRect(contour)
    out["area_ratio"] = area_ratio
    if area_ratio > 0.42 or (bw > 0.72 * 320 and bh > 0.48 * 240):
        out["reason"] = "blob_is_background"
        return out
    if bw < LIVE_MIN_BBOX_WIDTH_FRAC * 320 or bh < LIVE_MIN_BBOX_HEIGHT_FRAC * 240:
        out["reason"] = "too_small"
        return out
    if white_ratio < 0.028 or area_ratio < LIVE_PRESENT_AREA_MIN:
        out["reason"] = "low_occupancy"
        return out
    blob_mask = np.zeros((240, 320), np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, 255, -1)
    inside = blob_mask > 0
    mean_s = float(np.mean(S[inside])) if np.any(inside) else 255.0
    mean_tex = float(np.mean(tex[inside])) if np.any(inside) else 255.0
    mean_l = float(np.mean(L[inside])) if np.any(inside) else 0.0
    # Merged wood grain is more saturated and grainy than a fitting.
    if mean_s > 40.0 or mean_tex > 16.0 or mean_l < 160.0:
        out["reason"] = "wood"
        return out
    sx, sy = w / 320.0, h / 240.0
    out.update(
        {
            "found": True,
            "area_ratio": area_ratio,
            "bbox": [int(x * sx), int(y * sy), int((x + bw) * sx), int((y + bh) * sy)],
            "reason": "ok",
        }
    )
    return out


@dataclass
class TriggerState:
    phase: str = "EMPTY"
    present: bool = False
    motion: float = 0.0
    area_ratio: float = 0.0
    bbox: Optional[list] = None
    fire: bool = False
    label: str = "WAITING FOR PART"


@dataclass
class PresenceTrigger:
    present_min: float = LIVE_PRESENT_AREA_MIN
    empty_max: float = LIVE_EMPTY_AREA_MAX
    motion_stable: float = LIVE_MOTION_STABLE
    stable_needed: int = LIVE_STABLE_FRAMES
    empty_needed: int = LIVE_EMPTY_FRAMES
    _prev_small: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _bg: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _inspect_fp: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _stable: int = field(default=0, init=False)
    _empty: int = field(default=0, init=False)
    _phase: str = field(default="EMPTY", init=False)
    _warmup: int = field(default=0, init=False)

    def reset(self) -> None:
        self._prev_small = None
        self._inspect_fp = None
        self._stable = 0
        self._empty = 0
        self._phase = "EMPTY"

    def remember_inspect_frame(self, rgb: np.ndarray) -> None:
        self._inspect_fp = self._fingerprint(rgb)

    def mark_inspect_failed(self) -> None:
        """Occupancy saw a part but inspection did not stick — do not freeze on HOLD."""
        self._inspect_fp = None
        self._phase = "WAIT"
        self._stable = 0

    def set_empty_reference(self, rgb: np.ndarray) -> None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self._bg = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        self.reset()

    def update(self, rgb: np.ndarray) -> TriggerState:
        blob = detect_product_blob(rgb)
        area = float(blob["area_ratio"])
        white = float(blob.get("white_ratio") or 0.0)
        bbox = blob["bbox"] if blob.get("reason") == "ok" else None
        motion = self._motion(rgb)

        present = blob.get("reason") == "ok"
        empty = not present

        if self._bg is None and empty:
            self._warmup += 1
            if self._warmup >= 5:
                self.set_empty_reference(rgb)
        elif present:
            self._warmup = 0

        fire = False
        if empty:
            self._empty += 1
            self._stable = 0
            if self._empty >= 2:
                self._phase = "EMPTY"
        elif self._phase in ("EMPTY", "WAIT"):
            self._empty = 0
            if present and motion < self.motion_stable:
                self._stable += 1
            else:
                self._stable = 0
            self._phase = "WAIT"
            if self._stable >= self.stable_needed:
                self._phase = "FIRE"
                fire = True
                self._stable = 0
        else:
            self._empty = 0
            # Occupancy stays "present" when you swap fittings, so HOLD used to
            # freeze the first PASS for every later part. Re-inspect on change.
            # If inspection never stored a result, do not freeze on HOLD.
            if self._inspect_fp is None:
                self._phase = "WAIT"
                if present and motion < self.motion_stable:
                    self._stable += 1
                else:
                    self._stable = 0
                if self._stable >= self.stable_needed:
                    self._phase = "FIRE"
                    fire = True
                    self._stable = 0
            elif self._appearance_changed(rgb):
                self._phase = "WAIT"
                self._stable = 0
                self._inspect_fp = None
            else:
                self._phase = "HOLD"

        labels = {
            "EMPTY": "WAITING FOR PART",
            "WAIT": "PART DETECTED — HOLD STILL",
            "FIRE": "INSPECTING",
            "HOLD": "REMOVE OR SWAP FOR NEXT INSPECTION",
        }
        return TriggerState(
            phase=self._phase,
            present=present,
            motion=motion,
            area_ratio=white,
            bbox=bbox if present else None,
            fire=fire and present,
            label=labels.get(self._phase, self._phase),
        )

    def _fingerprint(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.resize(rgb, (80, 60), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _appearance_changed(self, rgb: np.ndarray) -> bool:
        now = self._fingerprint(rgb)
        return float(np.mean(np.abs(now - self._inspect_fp))) > 7.5

    def _motion(self, rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        if self._prev_small is None:
            self._prev_small = small
            return 999.0
        delta = np.mean(np.abs(small.astype(np.float32) - self._prev_small.astype(np.float32)))
        self._prev_small = small
        return float(delta)


def draw_station_overlay(
    rgb: np.ndarray,
    state: TriggerState,
    result_status: Optional[str] = None,
) -> np.ndarray:
    vis = rgb.copy()
    h, w = vis.shape[:2]
    # Occupancy box is presence only — never paint the whole fitting as the defect.
    box_color = (80, 180, 255)
    if state.bbox is not None:
        x0, y0, x1, y1 = state.bbox
        cv2.rectangle(vis, (x0, y0), (x1, y1), box_color, 3)
    banner = {
        "EMPTY": (180, 180, 180),
        "WAIT": (245, 197, 66),
        "FIRE": (80, 180, 255),
        "HOLD": (29, 214, 122)
        if result_status == "PASS"
        else ((255, 77, 79) if result_status == "FAIL" else (245, 197, 66)),
    }.get(state.phase, (180, 180, 180))
    bar_h = max(42, h // 16)
    cv2.rectangle(vis, (0, 0), (w, bar_h), (12, 16, 22), -1)
    text = state.label
    if result_status and state.phase == "HOLD":
        text = f"{result_status}   |   {state.label}"
    cv2.putText(
        vis,
        text,
        (16, bar_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        banner,
        2,
        cv2.LINE_AA,
    )
    return vis
