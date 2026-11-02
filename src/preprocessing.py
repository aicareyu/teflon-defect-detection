"""Stage 3: crop, rotate, scale, and center the isolated product."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import CANVAS_SIZE, LETTERBOX_FILL_RGB
from src.segmentation import _shape_stats


def _rotate_bundle(rgb: np.ndarray, masks: dict, angle_deg: float) -> tuple[np.ndarray, dict]:
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    out = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=LETTERBOX_FILL_RGB)
    rotated = {}
    for k, m in masks.items():
        rotated[k] = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return out, rotated


def _tight(rgb: np.ndarray, masks: dict, pad: int = 8) -> tuple[np.ndarray, dict]:
    prod = masks["product"]
    ys, xs = np.where(prod > 0)
    if len(xs) < 20:
        return rgb, masks
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = prod.shape[:2]
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(h, y1 + pad + 1), min(w, x1 + pad + 1)
    crop = rgb[y0:y1, x0:x1].copy()
    cropped = {k: v[y0:y1, x0:x1].copy() for k, v in masks.items()}
    return crop, cropped


def _letterbox(rgb: np.ndarray, masks: dict, size: int) -> tuple[np.ndarray, dict]:
    h, w = rgb.shape[:2]
    scale = size / float(max(h, w, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), LETTERBOX_FILL_RGB, np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    out_masks = {}
    for k, m in masks.items():
        rm = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
        c = np.zeros((size, size), np.uint8)
        c[y0 : y0 + nh, x0 : x0 + nw] = rm
        out_masks[k] = c
    return canvas, out_masks


def normalize_product(seg: dict, canvas: int = CANVAS_SIZE) -> dict:
    """
    Canonical canvas: rotate elongated parts upright, tight-crop, scale, center.
    Do not compare raw full-frame photos.
    """
    rgb = seg["cropped"].copy()
    masks = {
        "product": seg["product_mask"].copy(),
        "plastic": seg["plastic_mask"].copy(),
        "brass": seg["brass_mask"].copy(),
        "holes": seg.get("holes_mask", np.zeros(seg["product_mask"].shape[:2], np.uint8)).copy(),
    }
    ecc = float(seg.get("eccentricity") or 1.0)
    angle = float(seg.get("axis_angle_deg") or 0.0)
    rot = 0.0
    if ecc >= 1.35:
        rot = -(angle - 90.0)
        rgb, masks = _rotate_bundle(rgb, masks, rot)
    rgb, masks = _tight(rgb, masks)
    rgb, masks = _letterbox(rgb, masks, canvas)
    ecc2, ang2 = _shape_stats(masks["product"])
    plastic_n = max(int(cv2.countNonZero(masks["plastic"])), 1)
    brass_n = int(cv2.countNonZero(masks["brass"]))
    prod_n = max(int(cv2.countNonZero(masks["product"])), 1)
    hole_n = int(cv2.countNonZero(masks["holes"]))
    circ, solid = _contour_shape(masks["product"])
    out = {
        "rgb": rgb,
        "product_mask": masks["product"],
        "plastic_mask": masks["plastic"],
        "brass_mask": masks["brass"],
        "holes_mask": masks["holes"],
        "rotation_deg": float(rot),
        "eccentricity": float(ecc2),
        "axis_angle_deg": float(ang2),
        "brass_frac": brass_n / prod_n,
        "plastic_frac": plastic_n / prod_n,
        "hole_frac": hole_n / prod_n,
        "circularity": float(circ),
        "solidity": float(solid),
        "ok": prod_n > 80,
    }
    out.update(view_signature(out))
    return out


def _contour_shape(mask: np.ndarray) -> tuple[float, float]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    circ = float(4.0 * np.pi * area / max(peri * peri, 1e-6))
    hull = cv2.convexHull(c)
    ha = float(cv2.contourArea(hull))
    solid = float(area / max(ha, 1.0))
    return circ, solid


def view_signature(norm: dict) -> dict:
    """
    Coarse view class so a bore is not compared to a side profile.
    end_on: camera looking into a port (hole and/or brass fill).
    side: elongated body, openings not facing the camera.
    """
    ecc = float(norm.get("eccentricity") or 1.0)
    hole_frac = float(norm.get("hole_frac") or 0.0)
    brass_frac = float(norm.get("brass_frac") or 0.0)
    circ = float(norm.get("circularity") or 0.0)
    port = hole_frac >= 0.12 or brass_frac >= 0.10
    end_on = (ecc < 1.50 and hole_frac >= 0.14) or (ecc < 1.18 and port and circ >= 0.42)
    kind = "end_on" if end_on else "side"
    hsv = cv2.cvtColor(norm["rgb"], cv2.COLOR_RGB2HSV)
    plastic = norm["plastic_mask"] > 0
    sat_med = float(np.median(hsv[:, :, 1][plastic])) if np.count_nonzero(plastic) > 40 else 0.0
    return {"view_kind": kind, "sat_med": sat_med}
