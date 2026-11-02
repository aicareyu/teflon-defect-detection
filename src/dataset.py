"""Dataset discovery, analysis, and image-level labelling (GOOD / NOT GOOD)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.config import (
    CLASS_TO_IDX,
    GOOD_DIR,
    IMAGE_EXTENSIONS,
    NOT_GOOD_DIR,
)


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def collect_samples(
    good_dir: Path = GOOD_DIR,
    not_good_dir: Path = NOT_GOOD_DIR,
) -> Tuple[List[Path], np.ndarray, List[str]]:
    """
    Returns paths, integer labels, and string class names.

    good = 0, not_good = 1  (image-level labels only)
    """
    paths: List[Path] = []
    labels: List[int] = []
    names: List[str] = []

    for p in _list_images(good_dir):
        paths.append(p)
        labels.append(CLASS_TO_IDX["good"])
        names.append("good")
    for p in _list_images(not_good_dir):
        paths.append(p)
        labels.append(CLASS_TO_IDX["not_good"])
        names.append("not_good")

    if not paths:
        raise FileNotFoundError(
            "No images found. Place files in dataset/good and dataset/not_good "
            "(JPEG or PNG)."
        )
    return paths, np.asarray(labels, dtype=np.int64), names


def analyze_dataset(
    good_dir: Path = GOOD_DIR,
    not_good_dir: Path = NOT_GOOD_DIR,
) -> dict:
    """Summary statistics for the Dataset Analysis page."""
    good_files = _list_images(good_dir)
    ng_files = _list_images(not_good_dir)
    all_files = [("good", p) for p in good_files] + [("not_good", p) for p in ng_files]

    dims = []
    modes = Counter()
    samples = {"good": [], "not_good": []}
    errors = []

    for label, path in all_files:
        try:
            with Image.open(path) as im:
                w, h = im.size
                dims.append((w, h, path.name, label))
                modes[im.mode] += 1
                if len(samples[label]) < 8:
                    samples[label].append(
                        {
                            "path": str(path.as_posix()),
                            "name": path.name,
                            "width": w,
                            "height": h,
                            "mode": im.mode,
                        }
                    )
        except Exception as exc:  # noqa: BLE001 — surface corrupt files to the UI
            errors.append(f"{path.name}: {exc}")

    widths = [d[0] for d in dims]
    heights = [d[1] for d in dims]
    return {
        "n_good": len(good_files),
        "n_not_good": len(ng_files),
        "n_total": len(good_files) + len(ng_files),
        "good_files": [p.name for p in good_files],
        "not_good_files": [p.name for p in ng_files],
        "widths": widths,
        "heights": heights,
        "min_width": min(widths) if widths else None,
        "max_width": max(widths) if widths else None,
        "min_height": min(heights) if heights else None,
        "max_height": max(heights) if heights else None,
        "unique_sizes": sorted(set((d[0], d[1]) for d in dims)),
        "modes": dict(modes),
        "samples": samples,
        "errors": errors,
        "portrait": sum(1 for w, h, *_ in dims if h > w),
        "landscape": sum(1 for w, h, *_ in dims if w > h),
        "square": sum(1 for w, h, *_ in dims if w == h),
    }


def all_demo_images() -> List[Tuple[str, Path]]:
    """(display_label, path) pairs for Demo Mode."""
    items = []
    for p in _list_images(GOOD_DIR):
        items.append((f"GOOD / {p.name}", p))
    for p in _list_images(NOT_GOOD_DIR):
        items.append((f"NOT GOOD / {p.name}", p))
    return items
