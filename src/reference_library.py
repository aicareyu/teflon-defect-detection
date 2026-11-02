"""Stage 4: GOOD reference library. No training on the 15 photos."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import GOOD_DIR, IMAGE_EXTENSIONS, NOT_GOOD_DIR, ORB_FEATURES
from src.preprocessing import normalize_product
from src.segmentation import load_rgb, segment_product

_ORB = None
_EMBEDDER = None
_EMBEDDER_FAILED = False


def _orb() -> cv2.ORB:
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8)
    return _ORB


def plastic_histogram(rgb: np.ndarray, plastic: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], plastic, [18, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    return hist


def defect_signature(
    rgb: np.ndarray,
    plastic: np.ndarray,
    brass: np.ndarray | None = None,
    holes: np.ndarray | None = None,
) -> dict:
    """
    Lighting-relative marks: dark stain, yellow/brown spread, missing opening.
    Used to tell a clean GOOD photo from a NOT GOOD photo of a similar pose.
    """
    out = {
        "dark_frac": 0.0,
        "stain_frac": 0.0,
        "sat_p90": 0.0,
        "med_b": 0.0,
        "hole_frac": 0.0,
    }
    if brass is None:
        brass = np.zeros(plastic.shape[:2], np.uint8)
    sel = (plastic > 0) & (brass == 0)
    n = int(np.count_nonzero(sel))
    prod = max(int(cv2.countNonZero(plastic)) + int(cv2.countNonZero(brass)), 1)
    hole_n = int(cv2.countNonZero(holes)) if holes is not None else 0
    out["hole_frac"] = hole_n / float(prod)
    if n < 80:
        return out
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L, A, Bch = cv2.split(lab)
    S = hsv[:, :, 1]
    med_l = float(np.median(L[sel]))
    med_a = float(np.median(A[sel]))
    med_b = float(np.median(Bch[sel]))
    da = A.astype(np.float32) - med_a
    db = Bch.astype(np.float32) - med_b
    chroma = np.sqrt(da * da + db * db)
    dark = (L.astype(np.float32) <= med_l - 22.0) & sel & (L > 80)
    stain = (
        sel
        & (chroma >= 10.0)
        & (S.astype(np.float32) >= 14.0)
        & (L.astype(np.float32) <= med_l - 2.0)
        & (L > 70)
    )
    out.update(
        {
            "dark_frac": float(np.count_nonzero(dark)) / float(n),
            "stain_frac": float(np.count_nonzero(stain)) / float(n),
            "sat_p90": float(np.percentile(S[sel], 90)) / 255.0,
            "med_b": (med_b - 128.0) / 40.0,
            "hole_frac": hole_n / float(prod),
        }
    )
    return out


def describe_orb(gray: np.ndarray, mask: np.ndarray):
    kps, desc = _orb().detectAndCompute(gray, mask)
    return kps, desc


def embedding(rgb: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Disabled — loading ResNet on every library build made the station too slow."""
    return None


def _entry_from_norm(path: Path, norm: dict) -> dict:
    gray = cv2.cvtColor(norm["rgb"], cv2.COLOR_RGB2GRAY)
    kps, desc = describe_orb(gray, norm["product_mask"])
    return {
        "path": path,
        "name": path.name,
        "rgb": norm["rgb"],
        "gray": gray,
        "product_mask": norm["product_mask"],
        "plastic_mask": norm["plastic_mask"],
        "brass_mask": norm["brass_mask"],
        "holes_mask": norm["holes_mask"],
        "hist": plastic_histogram(norm["rgb"], norm["plastic_mask"]),
        "kps": kps,
        "desc": desc,
        "embed": None,
        "signature": defect_signature(
            norm["rgb"],
            norm["plastic_mask"],
            norm["brass_mask"],
            norm.get("holes_mask"),
        ),
        "eccentricity": float(norm["eccentricity"]),
        "brass_frac": float(norm["brass_frac"]),
        "plastic_frac": float(norm["plastic_frac"]),
        "hole_frac": float(norm.get("hole_frac") or 0.0),
        "circularity": float(norm.get("circularity") or 0.0),
        "solidity": float(norm.get("solidity") or 0.0),
        "view_kind": norm.get("view_kind") or "side",
        "sat_med": float(norm.get("sat_med") or 0.0),
    }


_SCHEMA = 5


def _load_folder(folder: Path, label: str) -> list[dict]:
    out = []
    if not folder.exists():
        return out
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rgb = load_rgb(path)
            seg = segment_product(rgb)
            if not seg.get("ok"):
                continue
            norm = normalize_product(seg)
            entry = _entry_from_norm(path, norm)
            entry["label"] = label
            out.append(entry)
        except Exception:
            continue
    return out


class ReferenceLibrary:
    def __init__(self):
        self.entries: list[dict] = []
        self.all_entries: list[dict] = []
        self._stamp = None
        self.reload()

    def _current_stamp(self):
        files = []
        for folder in (GOOD_DIR, NOT_GOOD_DIR):
            if folder.exists():
                files.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))
        return (_SCHEMA,) + tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)

    def reload(self, force: bool = False) -> None:
        stamp = self._current_stamp()
        if not force and stamp == self._stamp and self.all_entries:
            return
        goods = _load_folder(GOOD_DIR, "good")
        bads = _load_folder(NOT_GOOD_DIR, "not_good")
        self.entries = goods
        self.all_entries = goods + bads
        self._stamp = stamp

    def without(self, filename: str | None) -> list[dict]:
        self.reload()
        if not filename:
            return list(self.entries)
        key = Path(filename).name.lower()
        return [e for e in self.entries if e["name"].lower() != key]

    def all_without(self, filename: str | None) -> list[dict]:
        self.reload()
        if not filename:
            return list(self.all_entries)
        key = Path(filename).name.lower()
        skip = key in {"live.png", "frame.png"}
        if skip:
            return list(self.all_entries)
        return [e for e in self.all_entries if e["name"].lower() != key]


_LIB: ReferenceLibrary | None = None


def get_library() -> ReferenceLibrary:
    global _LIB
    if _LIB is None:
        _LIB = ReferenceLibrary()
    else:
        _LIB.reload()
    return _LIB
