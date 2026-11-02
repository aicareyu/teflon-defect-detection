"""GOOD appearance bank. Compare only normalized product crops, never raw frames."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.config import GOOD_DIR, IMAGE_EXTENSIONS
from src.segmentation import load_rgb, normalize_product, segment_product


def _list_good() -> List[Path]:
    if not GOOD_DIR.exists():
        return []
    return sorted(p for p in GOOD_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


class ReferenceManager:
    def __init__(self):
        self.entries: List[dict] = []

    def rebuild(self) -> int:
        self.entries = []
        for path in _list_good():
            rgb = load_rgb(path)
            seg = segment_product(rgb)
            if not seg["ok"]:
                continue
            crop, mask, _ = normalize_product(
                seg["cropped"], seg["white_mask"] | seg["brass_mask"], seg["eccentricity"], seg["axis_angle_deg"]
            )
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            white_frac = cv2.countNonZero(seg["white_mask"]) / max(gray.size, 1)
            brass_frac = cv2.countNonZero(seg["brass_mask"]) / max(gray.size, 1)
            self.entries.append(
                {
                    "name": path.name,
                    "gray": gray,
                    "ecc": seg["eccentricity"],
                    "brass_frac": brass_frac,
                    "white_frac": white_frac,
                }
            )
        return len(self.entries)

    def match(self, ecc: float, brass_frac: float, white_frac: float) -> Optional[np.ndarray]:
        if not self.entries:
            return None
        best, best_d = None, 1e9
        for e in self.entries:
            d = (
                abs(e["ecc"] - ecc) / 3.0
                + abs(e["brass_frac"] - brass_frac) * 4.0
                + abs(e["white_frac"] - white_frac) * 2.0
            )
            if d < best_d:
                best_d, best = d, e
        # Reject a badly mismatched view (e.g. circular bore vs long T).
        if best is None or best_d > 1.8:
            return None
        return best["gray"]


_BANK: Optional[ReferenceManager] = None


def get_reference_manager() -> ReferenceManager:
    global _BANK
    if _BANK is None:
        _BANK = ReferenceManager()
        _BANK.rebuild()
    return _BANK
