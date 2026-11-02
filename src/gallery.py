"""
Nearest-neighbour classifier on ResNet-18 embeddings.

Training photos (granite, mixed shapes) are weak for the live Femto station.
Live teach samples are stored in the same gallery with a high vote weight so
one 'This is GOOD' click immediately corrects the station.
"""

from __future__ import annotations

import numpy as np

from src.config import DISPLAY_NAMES, GALLERY_PATH, USE_PRODUCT_CROP
from src.dataset import collect_samples
from src.inspection import decide_status, format_result_block
from src.model import ResNet18Encoder, extract_embeddings, get_device
from src.preprocess import imagenet_tensor, prepare_from_path, prepare_rgb

LIVE_VOTE_WEIGHT = 10.0
K_NEIGHBORS = 5

_STATE = {"encoder": None, "device": None, "X": None, "y": None, "live": None, "names": None}


def _encoder():
    if _STATE["encoder"] is None:
        device = get_device()
        enc = ResNet18Encoder()
        enc.to(device)
        enc.eval()
        _STATE["encoder"] = enc
        _STATE["device"] = device
    return _STATE["encoder"], _STATE["device"]


def embed_rgb(rgb: np.ndarray, use_crop: bool = USE_PRODUCT_CROP):
    display, net, info = prepare_rgb(rgb, use_crop=use_crop)
    tensor = imagenet_tensor(net)
    enc, device = _encoder()
    X, _ = extract_embeddings(enc, tensor, device, batch_size=1)
    vec = X[0].astype(np.float32)
    n = np.linalg.norm(vec) + 1e-8
    return vec / n, display, info


def _load_disk() -> bool:
    if not GALLERY_PATH.exists():
        return False
    data = np.load(GALLERY_PATH, allow_pickle=True)
    _STATE["X"] = data["X"].astype(np.float32)
    _STATE["y"] = data["y"].astype(np.int64)
    _STATE["live"] = data["live"].astype(np.int64)
    _STATE["names"] = data["names"]
    return len(_STATE["y"]) > 0


def _save_disk() -> None:
    GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        GALLERY_PATH,
        X=_STATE["X"],
        y=_STATE["y"],
        live=_STATE["live"],
        names=_STATE["names"],
    )


def build_gallery_from_dataset() -> dict:
    paths, y, _ = collect_samples()
    enc, device = _encoder()
    vecs = []
    names = []
    for p, label in zip(paths, y):
        _orig, _disp, net, _info = prepare_from_path(p, use_crop=USE_PRODUCT_CROP)
        tensor = imagenet_tensor(net)
        X, _ = extract_embeddings(enc, tensor, device, batch_size=1)
        v = X[0].astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        vecs.append(v)
        names.append(p.name)
    _STATE["X"] = np.stack(vecs, axis=0)
    _STATE["y"] = np.asarray(y, dtype=np.int64)
    _STATE["live"] = np.zeros(len(y), dtype=np.int64)
    _STATE["names"] = np.asarray(names)
    _save_disk()
    return {"n": len(y), "n_good": int((y == 0).sum()), "n_not_good": int((y == 1).sum())}


def ensure_gallery() -> None:
    if _STATE["X"] is not None:
        return
    if _load_disk():
        return
    build_gallery_from_dataset()


def add_live_example(rgb: np.ndarray, label: str, filename: str) -> None:
    ensure_gallery()
    idx = 0 if label == "good" else 1
    vec, _display, _info = embed_rgb(rgb)
    _STATE["X"] = np.vstack([_STATE["X"], vec[None, :]])
    _STATE["y"] = np.concatenate([_STATE["y"], [idx]])
    _STATE["live"] = np.concatenate([_STATE["live"], [1]])
    _STATE["names"] = np.concatenate([_STATE["names"], [filename]])
    _save_disk()


def predict_embedding(vec: np.ndarray, threshold: float) -> dict:
    ensure_gallery()
    X = _STATE["X"]
    y = _STATE["y"]
    live = _STATE["live"]
    sims = X @ vec
    weights = np.where(live == 1, LIVE_VOTE_WEIGHT, 1.0).astype(np.float32)
    k = min(K_NEIGHBORS, len(y))
    top = np.argsort(sims)[-k:]
    score = {0: 0.0, 1: 0.0}
    for i in top:
        score[int(y[i])] += float(max(sims[i], 0.0) * weights[i])
    total = score[0] + score[1] + 1e-8
    p0 = score[0] / total
    p1 = score[1] / total
    pred_idx = 0 if p0 >= p1 else 1
    confidence = float(max(p0, p1))
    prediction = DISPLAY_NAMES[pred_idx]
    status = decide_status(pred_idx, confidence, threshold)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "good_probability": float(p0),
        "not_good_probability": float(p1),
        "predicted_index": pred_idx,
        "status": status,
        "threshold": float(threshold),
        "n_gallery": int(len(y)),
        "n_live": int(live.sum()),
    }


def classify_rgb(rgb: np.ndarray, filename: str, threshold: float, use_crop: bool = True) -> dict:
    vec, display, info = embed_rgb(rgb, use_crop=use_crop)
    out = predict_embedding(vec, threshold)
    out.update(
        {
            "filename": filename,
            "original_rgb": rgb,
            "display_rgb": display,
            "crop_info": info,
            "result_text": format_result_block(out["status"], out["prediction"], out["confidence"]),
        }
    )
    return out


def gallery_counts() -> dict:
    ensure_gallery()
    y = _STATE["y"]
    live = _STATE["live"]
    return {
        "n": int(len(y)),
        "n_good": int((y == 0).sum()),
        "n_not_good": int((y == 1).sum()),
        "n_live": int(live.sum()),
    }
