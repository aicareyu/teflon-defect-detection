"""Overlay helpers for the inspection dashboard."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def colorize_anomaly(anomaly: np.ndarray) -> np.ndarray:
    if anomaly is None or anomaly.size == 0:
        return np.zeros((8, 8, 3), np.uint8)
    mx = float(anomaly.max()) if anomaly.size else 1.0
    scale = 1.0 / max(mx, 0.15)
    u8 = np.uint8(np.clip(anomaly * scale * 255.0, 0, 255))
    heat = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def overlay_defects(rgb: np.ndarray, blobs: List[dict]) -> np.ndarray:
    vis = rgb.copy()
    h, w = vis.shape[:2]
    for b in blobs:
        x0, y0, x1, y1 = [int(v) for v in b["bbox"]]
        cnt = b.get("contour")
        if cnt is not None and len(cnt) >= 5:
            mask = np.zeros((h, w), np.uint8)
            xs = np.clip(cnt[:, 0], 0, w - 1)
            ys = np.clip(cnt[:, 1], 0, h - 1)
            mask[ys, xs] = 255
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            tint = vis.astype(np.float32)
            tint[mask > 0] = tint[mask > 0] * 0.45 + np.array([255.0, 40.0, 40.0]) * 0.55
            vis = np.clip(tint, 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(vis, contours, -1, (255, 40, 40), 2)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (220, 30, 30), 2)
        cv2.putText(
            vis,
            b.get("label", "SUSPECTED DEFECT"),
            (x0, max(18, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 50, 50),
            1,
            cv2.LINE_AA,
        )
    return vis


def draw_bbox(rgb: np.ndarray, bbox, color=(40, 200, 90)) -> np.ndarray:
    vis = rgb.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox]
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
    return vis


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, color=(0, 200, 255), alpha=0.35) -> np.ndarray:
    vis = rgb.copy()
    tint = np.zeros_like(vis)
    tint[:, :] = color
    m = mask > 0
    vis[m] = (vis[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
    return vis


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    if mask is None or getattr(mask, "size", 0) == 0:
        return np.zeros((8, 8, 3), np.uint8)
    if mask.ndim == 3:
        return mask
    m = mask
    if m.dtype != np.uint8:
        mx = float(m.max()) if m.size else 1.0
        m = np.uint8(np.clip(m / max(mx, 1e-6), 0, 1) * 255) if mx <= 1.5 else np.clip(m, 0, 255).astype(np.uint8)
    elif int(m.max()) <= 1:
        m = np.uint8(m * 255)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
