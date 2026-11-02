"""Stage 1: isolate the fitting and split white plastic vs brass vs background."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def load_rgb(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.uint8((labels == idx) * 255)


def segment_product(rgb: np.ndarray) -> dict:
    """
    Return mask, bbox, cropped product, white/brass masks.

    White/cream plastic: high L, low saturation.
    Brass insert: yellow-gold HSV.
    Adaptive: if HSV white is weak, fall back to Otsu on L.
    """
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    white_hsv = cv2.inRange(hsv, (0, 0, 135), (180, 82, 255))
    # Cream plastic is slightly yellow in LAB (positive b*).
    white_lab = np.uint8(((L > 130) & (np.abs(A.astype(np.int16) - 128) < 18) & (B > 118)) * 255)
    white = cv2.bitwise_or(white_hsv, white_lab)

    brass = cv2.inRange(hsv, (10, 80, 80), (38, 255, 255))
    # Drop brass-colored table veins far from the part: keep later via proximity.

    if cv2.countNonZero(white) < 0.004 * h * w:
        blur = cv2.GaussianBlur(L, (7, 7), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white = otsu

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=1)

    white_d = cv2.dilate(white, kernel, iterations=2)
    brass_near = cv2.bitwise_and(brass, white_d)
    product = cv2.bitwise_or(white, brass_near)
    product = cv2.morphologyEx(product, cv2.MORPH_CLOSE, kernel, iterations=2)
    product = _largest_component(product)
    raw_product = product.copy()

    ys, xs = np.where(product > 0)
    if len(xs) >= 80:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = 12
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
        bbox = [x0, y0, x1, y1]
        crop = rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
        crop_product = product[y0 : y1 + 1, x0 : x1 + 1]
        crop_raw = raw_product[y0 : y1 + 1, x0 : x1 + 1]
        crop_white = cv2.bitwise_and(white[y0 : y1 + 1, x0 : x1 + 1], crop_product)
        crop_brass = cv2.bitwise_and(brass[y0 : y1 + 1, x0 : x1 + 1], crop_product)
    else:
        bbox = [0, 0, w - 1, h - 1]
        crop = rgb.copy()
        crop_product = product
        crop_raw = raw_product
        crop_white = white
        crop_brass = cv2.bitwise_and(brass, product)

    ecc, angle = _shape_stats(product)
    holes = _interior_holes(crop_product)
    crop_white = cv2.bitwise_and(crop_white, cv2.bitwise_not(holes))
    full_white = cv2.bitwise_and(white, product)
    full_brass = cv2.bitwise_and(brass, product)
    background = cv2.bitwise_not(product)
    return {
        "mask": product,
        "bbox": bbox,
        "cropped": crop,
        "cropped_product": crop,
        "product_mask": crop_product,
        "raw_product_mask": crop_raw,
        "white_mask": crop_white,
        "plastic_mask": crop_white,
        "brass_mask": crop_brass,
        "background_mask": background,
        "holes_mask": holes,
        "full_white": full_white,
        "full_brass": full_brass,
        "eccentricity": ecc,
        "axis_angle_deg": angle,
        "ok": cv2.countNonZero(product) > 80,
    }


def _interior_holes(product_mask: np.ndarray) -> np.ndarray:
    """Dark bores / openings inside the silhouette — not surface defects."""
    if product_mask is None or product_mask.size == 0:
        return np.zeros((1, 1), np.uint8)
    pad = cv2.copyMakeBorder(product_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ph, pw = pad.shape[:2]
    ff = np.zeros((ph + 2, pw + 2), np.uint8)
    flood = pad.copy()
    cv2.floodFill(flood, ff, (0, 0), 255)
    return cv2.bitwise_not(flood)[1:-1, 1:-1]


def _shape_stats(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return 1.0, 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    cov = np.cov(pts.T)
    eig, vec = np.linalg.eigh(cov)
    eig = np.maximum(eig, 1e-6)
    ecc = float(np.sqrt(eig[1] / eig[0]))
    axis = vec[:, 1]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return ecc, angle


def normalize_product(crop: np.ndarray, product_mask: np.ndarray, ecc: float, angle: float):
    """Rotate elongated parts onto a vertical axis. Skip near-circular views."""
    if ecc < 1.35 or crop.size == 0:
        return crop, product_mask, 0.0
    h, w = crop.shape[:2]
    # Make major axis vertical.
    rot = -(angle - 90.0)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, 1.0)
    out = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(20, 20, 20))
    m = cv2.warpAffine(product_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return out, m, rot
