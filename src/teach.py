"""Save a live frame into dataset/good or dataset/not_good and refit."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from src.config import GOOD_DIR, NOT_GOOD_DIR
from src.retrain import retrain_embedding_classifier


def save_live_example(rgb, label: str) -> Path:
    if label not in {"good", "not_good"}:
        raise ValueError("label must be good or not_good")
    folder = GOOD_DIR if label == "good" else NOT_GOOD_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"live-{stamp}.jpeg"
    Image.fromarray(rgb).save(path, quality=95)
    return path


def teach_and_retrain(rgb, label: str) -> dict:
    path = save_live_example(rgb, label)
    stats = retrain_embedding_classifier()
    stats["saved"] = path.as_posix()
    stats["label"] = label
    return stats
