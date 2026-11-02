"""
Image preprocessing for industrial inspection.

Goals
-----
1. Keep the physical defect intact (no aggressive warping / colour jitter).
2. Reduce background leakage with an optional foreground crop.
3. Preserve aspect ratio via letterbox resize (handles portrait + landscape).
4. Apply ImageNet normalisation expected by the pretrained backbone.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    CROP_MAX_AREA_RATIO,
    CROP_MIN_AREA_RATIO,
    CROP_PAD_RATIO,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LETTERBOX_FILL_RGB,
    USE_PRODUCT_CROP,
)


def load_rgb(image_path) -> np.ndarray:
    """Load JPEG/PNG (and similar) as RGB uint8 array. Handles EXIF orientation."""
    with Image.open(image_path) as pil:
        pil = pil.convert("RGB")
        return np.array(pil)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect_product_blob(rgb: np.ndarray) -> dict:
    """
    Find the fitting (white plastic + nearby brass), not the work surface.

    Otsu-on-brightness was matching the whole wooden table (~99% occupancy)
    and blocked auto-trigger. White PP is high-value / low-saturation; brass
    is kept only where it touches that white body.
    """
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
    bgr = rgb_to_bgr(rgb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # White / cream plastic body.
    white = cv2.inRange(hsv, (0, 0, 160), (180, 55, 255))
    # Brass / gold metal next to the plastic (wood is excluded unless adjacent).
    brass = cv2.inRange(hsv, (8, 55, 70), (38, 255, 255))
    white_ratio = float(cv2.countNonZero(white)) / float(h * w)
    out["white_ratio"] = white_ratio

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    white_d = cv2.dilate(white, kernel, iterations=2)
    near_brass = cv2.bitwise_and(brass, white_d)
    mask = cv2.bitwise_or(white, near_brass)

    if white_ratio < 0.004:
        # Studio shots on dark granite: fall back to brightness blob.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, bright = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_or(bright, brass)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        out["reason"] = "no_contour"
        return out
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    area_ratio = area / float(h * w)
    if area_ratio > 0.70:
        out["reason"] = "blob_is_background"
        return out
    x, y, bw, bh = cv2.boundingRect(contour)
    keep = np.zeros_like(mask)
    cv2.drawContours(keep, [contour], -1, 255, thickness=cv2.FILLED)
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    out.update(
        {
            "found": True,
            "area_ratio": area_ratio,
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "reason": "ok",
            "mask": keep,
        }
    )
    return out


def crop_product(
    rgb: np.ndarray,
    pad_ratio: float = CROP_PAD_RATIO,
) -> Tuple[np.ndarray, dict]:
    """
    Optional automatic foreground crop.

    The fittings are bright plastic/metal on a dark work surface. We build a
    brightness + saturation mask, keep the largest blob, and pad the box so
    that defects on the rim are not cut off.

    If the mask is unreliable, the original image is returned unchanged.
    This is a *debug aid*, not a precise segmentation of the defect.
    """
    blob = detect_product_blob(rgb)
    info = {
        "applied": False,
        "reason": blob["reason"],
        "bbox": blob["bbox"],
        "area_ratio": blob["area_ratio"],
    }
    if rgb is None or rgb.size == 0:
        return rgb, info
    if not blob["found"]:
        return rgb, info

    h, w = rgb.shape[:2]
    area_ratio = blob["area_ratio"]
    if area_ratio < CROP_MIN_AREA_RATIO:
        info["reason"] = "blob_too_small"
        return rgb, info
    if area_ratio > CROP_MAX_AREA_RATIO:
        info["reason"] = "blob_fills_frame"
        return rgb, info

    x, y, x1b, y1b = blob["bbox"]
    bw, bh = x1b - x, y1b - y
    pad_x = int(bw * pad_ratio) + 8
    pad_y = int(bh * pad_ratio) + 8
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x1b + pad_x)
    y1 = min(h, y1b + pad_y)

    if (x1 - x0) < 16 or (y1 - y0) < 16:
        info["reason"] = "bbox_too_small"
        return rgb, info

    cropped = rgb[y0:y1, x0:x1].copy()
    info.update(
        {
            "applied": True,
            "reason": "ok",
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
        }
    )
    return cropped, info


def letterbox(
    rgb: np.ndarray,
    size: int = IMAGE_SIZE,
    fill_rgb: Tuple[int, int, int] = LETTERBOX_FILL_RGB,
) -> np.ndarray:
    """Resize with unchanged aspect ratio and pad to a square canvas."""
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot letterbox an empty image.")
    scale = size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:, :] = fill_rgb
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def prepare_rgb(
    rgb: np.ndarray,
    use_crop: bool = USE_PRODUCT_CROP,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns (display_rgb, network_rgb_224, crop_info).

    display_rgb is the (optionally cropped) product image used for overlays.
    network_rgb_224 is letterboxed to IMAGE_SIZE.
    """
    crop_info = {"applied": False, "reason": "disabled", "bbox": None, "area_ratio": 0.0}
    working = rgb
    if use_crop:
        working, crop_info = isolate_and_crop(rgb)
    net = letterbox(working, IMAGE_SIZE)
    return working, net, crop_info


def isolate_and_crop(rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Paint the work surface a neutral grey, then crop the fitting.

    Live Femto frames are on wood; training photos are on granite. If the
    background stays in the tensor, every live part looks 'not good'.
    """
    blob = detect_product_blob(rgb)
    isolated = rgb.copy()
    mask = blob.get("mask")
    if mask is not None:
        isolated[mask == 0] = (42, 42, 42)
    cropped, info = crop_product(isolated)
    info["isolated"] = mask is not None
    return cropped, info


def imagenet_tensor(rgb_224: np.ndarray) -> torch.Tensor:
    """RGB uint8 HxWxC -> float tensor 1x3xHxW with ImageNet stats."""
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return t(Image.fromarray(rgb_224)).unsqueeze(0)


def build_train_augment():
    """
    Mild, physically plausible augmentation.

    Avoids colour inversion, large crops that drop the part, and geometric
    warps that would invent or hide a real defect.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
                fill=LETTERBOX_FILL_RGB,
            ),
            transforms.ColorJitter(
                brightness=0.18, contrast=0.18, saturation=0.08, hue=0.02
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def pil_from_rgb(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb)


def prepare_from_path(
    image_path,
    use_crop: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a file and return original, cropped-or-original, letterboxed, crop_info."""
    if use_crop is None:
        use_crop = USE_PRODUCT_CROP
    original = load_rgb(image_path)
    display, net, info = prepare_rgb(original, use_crop=use_crop)
    return original, display, net, info
