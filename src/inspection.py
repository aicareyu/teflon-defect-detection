"""Orchestrator: isolate → normalize → best GOOD view → local residual → decision."""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.config import DEFAULT_PARAMS, LOGS_DIR, SMALL_DATA_WARNING, STATUS_FAIL, STATUS_PASS
from src.defect_detector import decide, detect_local_anomalies
from src.preprocessing import normalize_product
from src.presence import detect_product_blob
from src.reference_library import defect_signature, describe_orb, get_library, plastic_histogram
from src.registration import align_to_reference
from src.segmentation import load_rgb, segment_product
from src.view_matching import find_best_reference
from src.visualization import colorize_anomaly, overlay_defects


def make_inspection_id() -> str:
    now = datetime.now()
    return f"INSP-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"


def _query_features(norm: dict) -> dict:
    gray = cv2.cvtColor(norm["rgb"], cv2.COLOR_RGB2GRAY)
    kps, desc = describe_orb(gray, norm["product_mask"])
    q = dict(norm)
    q["gray"] = gray
    q["kps"] = kps
    q["desc"] = desc
    q["hist"] = plastic_histogram(norm["rgb"], norm["plastic_mask"])
    q["signature"] = defect_signature(
        norm["rgb"],
        norm["plastic_mask"],
        norm["brass_mask"],
        norm.get("holes_mask"),
    )
    q["embed"] = None
    return q


def _empty_live(rgb: np.ndarray, params: Dict, filename: str, t0: float) -> dict:
    h, w = rgb.shape[:2]
    blank = np.zeros((max(h, 8), max(w, 8), 3), np.uint8)
    return {
        "status": "WAITING",
        "headline": "NO PART IN VIEW",
        "why": "No cream plastic fitting in the frame. An empty table is not a PASS.",
        "score": 0.0,
        "match_score": 0.0,
        "local_score": 0.0,
        "ref_diff": 0.0,
        "inspection_id": make_inspection_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_ms": (time.perf_counter() - t0) * 1000.0,
        "filename": filename,
        "original": rgb,
        "working": rgb,
        "product_roi": blank,
        "anomaly_rgb": blank,
        "overlay": blank,
        "best_reference_rgb": blank,
        "aligned_rgb": blank,
        "blobs": [],
        "n_blobs": 0,
        "best_reference": None,
        "debug": {"seg_ok": False, "engine": "ref-poc-v3"},
        "engine": "ref-poc-v3",
        "disclaimer": SMALL_DATA_WARNING,
        "params": params,
        "no_part": True,
    }


def inspect_image(rgb: np.ndarray, params: Optional[Dict] = None, filename: str = "frame.png") -> dict:
    t0 = time.perf_counter()
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    live_mode = bool(p.get("live_mode")) or str(filename).lower().startswith("live")
    p["live_mode"] = live_mode
    h, w = rgb.shape[:2]
    mx = max(h, w)
    if mx > 960:
        s = 960.0 / float(mx)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    if live_mode and detect_product_blob(rgb).get("reason") != "ok":
        return _empty_live(rgb, p, filename, t0)
    if live_mode and not p.get("skip_live_save"):
        try:
            import shutil

            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            dest = LOGS_DIR / "last_live.png"
            prev = LOGS_DIR / "prev_live.png"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if dest.exists():
                shutil.copy2(dest, prev)
            cv2.imwrite(str(dest), bgr)
        except Exception:
            pass

    seg = segment_product(rgb)
    if not seg.get("ok"):
        return _empty_live(rgb, p, filename, t0)
    norm = normalize_product(seg)
    query = _query_features(norm)
    lib = get_library()
    goods = lib.without(None if live_mode else filename)
    match = find_best_reference(query, goods)

    aligned = None
    det = {
        "anomaly_map": np.zeros(norm["rgb"].shape[:2], np.float32),
        "candidate_mask": np.zeros(norm["rgb"].shape[:2], np.uint8),
        "interior_mask": np.zeros(norm["rgb"].shape[:2], np.uint8),
        "blobs": [],
        "best": None,
        "local_score": 0.0,
        "mean_ssim": 1.0,
        "ref_diff": 1.0,
        "concentrated": 0.0,
        "search_n": 0,
        "align_ok": False,
    }
    view_compatible = bool(match.get("compatible")) and match.get("ref") is not None
    p["view_compatible"] = view_compatible
    if match.get("ref") is not None:
        aligned = align_to_reference(query, match["ref"])
        for k in ("view_kind", "circularity", "hole_frac", "eccentricity", "solidity", "brass_frac"):
            aligned[k] = query.get(k)
        det = detect_local_anomalies(aligned, match["ref"], p)
    else:
        aligned = {
            "rgb": query["rgb"],
            "product_mask": query["product_mask"],
            "plastic_mask": query["plastic_mask"],
            "brass_mask": query["brass_mask"],
            "holes_mask": query.get("holes_mask"),
            "method": "none",
            "inliers": 0,
            "ecc_corr": 0.0,
            "view_kind": query.get("view_kind"),
            "circularity": query.get("circularity"),
            "hole_frac": query.get("hole_frac"),
            "eccentricity": query.get("eccentricity"),
        }
        det = detect_local_anomalies(aligned, None, p)

    prefer = [
        b
        for b in det.get("blobs") or []
        if b.get("kind") in ("crushed_opening", "color_spread", "black_mark", "broken_opening")
    ]
    use = prefer[0] if prefer else (det.get("best") or {})
    status, why = decide(
        det["local_score"] if not prefer else float(use.get("anomaly_score") or 0) * 1.2,
        float(match["score"]),
        det["ref_diff"],
        p,
        align_ok=bool(det.get("align_ok")),
        best_kind=use.get("kind"),
        best_area=int(use.get("area") or 0),
        best_contrast=float(use.get("local_contrast") or 0.0),
        own_contrast=float(use.get("own_contrast") or 0.0),
        view_kind=str(query.get("view_kind") or "side"),
    )
    vis_blobs = []
    if status == STATUS_FAIL and use:
        vis_blobs = [use]
    overlay_src = aligned["rgb"] if aligned is not None else norm["rgb"]
    overlay = overlay_defects(overlay_src, vis_blobs[:1])
    heat = colorize_anomaly(det["anomaly_map"])
    ref_rgb = match["ref"]["rgb"] if match.get("ref") is not None else np.zeros_like(norm["rgb"])

    if status == STATUS_FAIL:
        headline = "DEFECTIVE"
    else:
        headline = "GOOD"

    debug = {
        "seg_ok": True,
        "eccentricity": float(norm["eccentricity"]),
        "rotation_deg": float(norm["rotation_deg"]),
        "brass_frac": float(norm["brass_frac"]),
        "align_method": aligned["method"] if aligned else "none",
        "align_inliers": aligned["inliers"] if aligned else 0,
        "ecc_corr": aligned.get("ecc_corr") if aligned else 0,
        "align_ok": det.get("align_ok"),
        "view_kind": query.get("view_kind"),
        "view_compatible": view_compatible,
        "circularity": query.get("circularity"),
        "hole_frac": query.get("hole_frac"),
        "signature": query.get("signature"),
        "match": {
            k: match[k]
            for k in ("score", "hist", "shape", "orb", "embed", "sig", "name", "compatible", "label")
            if k in match
        },
        "mean_ssim": det["mean_ssim"],
        "local_score": det["local_score"],
        "ref_diff": det["ref_diff"],
        "concentrated": det["concentrated"],
        "product_mask": norm["product_mask"],
        "plastic_mask": norm["plastic_mask"],
        "brass_mask": norm["brass_mask"],
        "normalized": norm["rgb"],
        "aligned_rgb": overlay_src,
        "candidate_mask": det["candidate_mask"],
        "interior_mask": det["interior_mask"],
        "anomaly_map": det["anomaly_map"],
        "bbox": seg["bbox"],
        "engine": "ref-poc-v3",
    }
    return {
        "status": status,
        "headline": headline,
        "why": why,
        "score": 0.0 if status == STATUS_PASS else float(det["local_score"]),
        "match_score": float(match["score"]),
        "local_score": float(det["local_score"]),
        "ref_diff": float(det["ref_diff"]),
        "mean_ssim": float(det["mean_ssim"]),
        "inspection_id": make_inspection_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_ms": (time.perf_counter() - t0) * 1000.0,
        "filename": filename,
        "original": rgb,
        "working": rgb,
        "product_roi": seg["cropped"],
        "anomaly_rgb": heat,
        "overlay": overlay,
        "best_reference_rgb": ref_rgb,
        "aligned_rgb": overlay_src,
        "blobs": det["blobs"],
        "n_blobs": len(det["blobs"]),
        "best_reference": match.get("name"),
        "ranked_references": match.get("ranked") or [],
        "debug": debug,
        "engine": "ref-poc-v3",
        "disclaimer": SMALL_DATA_WARNING,
        "params": p,
        "no_part": False,
    }


def inspect_path(path, params: Optional[Dict] = None) -> dict:
    path = Path(path)
    rgb = load_rgb(path)
    return inspect_image(rgb, params=params, filename=path.name)


def append_log(result: dict, source: str = "manual") -> None:
    import csv

    from src.config import INSPECTION_LOG_CSV

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not INSPECTION_LOG_CSV.exists()
    with INSPECTION_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(
                ["id", "time", "file", "status", "score", "match", "reference", "source"]
            )
        w.writerow(
            [
                result.get("inspection_id"),
                result.get("timestamp"),
                result.get("filename"),
                result.get("status"),
                f"{float(result.get('score') or 0):.4f}",
                f"{float(result.get('match_score') or 0):.4f}",
                result.get("best_reference") or "",
                source,
            ]
        )
