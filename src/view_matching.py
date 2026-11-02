"""Stage 5: pick the most similar GOOD reference view. No class training."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))


def _sig_sim(q: dict, r: dict) -> float:
    """How similar the stain / dark-mark / hole signature is."""
    qs = q.get("signature") or {}
    rs = r.get("signature") or {}
    scales = {
        "dark_frac": 0.07,
        "stain_frac": 0.08,
        "sat_p90": 0.14,
        "med_b": 0.40,
        "hole_frac": 0.12,
    }
    dist = 0.0
    for key, scale in scales.items():
        dist += abs(float(qs.get(key) or 0.0) - float(rs.get(key) or 0.0)) / scale
    return float(np.clip(1.0 - 0.22 * dist, 0.0, 1.0))


def _shape_sim(q: dict, r: dict) -> float:
    de = abs(float(q.get("eccentricity", 1.0)) - float(r.get("eccentricity", 1.0)))
    db = abs(float(q.get("brass_frac", 0.0)) - float(r.get("brass_frac", 0.0)))
    dp = abs(float(q.get("plastic_frac", 0.0)) - float(r.get("plastic_frac", 0.0)))
    return float(np.clip(1.0 - 0.18 * de - 1.6 * db - 0.8 * dp, 0.0, 1.0))


def _orb_score(q_desc, r_desc) -> float:
    if q_desc is None or r_desc is None or len(q_desc) < 8 or len(r_desc) < 8:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(q_desc, r_desc, k=2)
    good = []
    for item in pairs:
        if len(item) < 2:
            continue
        a, b = item
        if a.distance < 0.75 * b.distance:
            good.append(a)
    denom = float(min(len(q_desc), len(r_desc), 80))
    return float(min(len(good) / max(denom, 1.0), 1.0))


def _embed_cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def views_compatible(query: dict, ref: dict) -> bool:
    """Reject a GOOD photo that is clearly a different camera view."""
    qk = query.get("view_kind") or "side"
    rk = ref.get("view_kind") or "side"
    if qk != rk:
        return False
    if abs(float(query.get("eccentricity") or 1) - float(ref.get("eccentricity") or 1)) > 0.40:
        return False
    if abs(float(query.get("circularity") or 0) - float(ref.get("circularity") or 0)) > 0.28:
        return False
    if abs(float(query.get("solidity") or 0) - float(ref.get("solidity") or 0)) > 0.28:
        return False
    # Side views: brass amount distinguishes a hex adapter from a plastic T.
    # End-on views: brass vs a dark bore is the same port, not a different pose.
    if qk == "side":
        if abs(float(query.get("brass_frac") or 0) - float(ref.get("brass_frac") or 0)) > 0.18:
            return False
    return True


def score_reference(query: dict, ref: dict) -> dict:
    hist = max(0.0, _hist_corr(query.get("hist"), ref.get("hist")))
    shape = _shape_sim(query, ref)
    orb = _orb_score(query.get("desc"), ref.get("desc"))
    emb = _embed_cos(query.get("embed"), ref.get("embed"))
    compatible = views_compatible(query, ref)
    sig = _sig_sim(query, ref)
    if query.get("embed") is not None and ref.get("embed") is not None:
        total = 0.24 * emb + 0.18 * orb + 0.16 * hist + 0.12 * shape + 0.30 * sig
    else:
        total = 0.38 * orb + 0.30 * hist + 0.24 * shape + 0.08 * sig
    if not compatible:
        total *= 0.12
    return {
        "score": float(total),
        "hist": float(hist),
        "shape": float(shape),
        "orb": float(orb),
        "embed": float(emb),
        "sig": float(sig),
        "compatible": bool(compatible),
        "name": ref["name"],
        "path": ref["path"],
    }


def find_nearest_example(query: dict, entries: list[dict]) -> dict:
    """Closest dataset photo (GOOD or NOT GOOD). Used for the live station decision."""
    blank = {
        "score": 0.0,
        "hist": 0.0,
        "shape": 0.0,
        "orb": 0.0,
        "embed": 0.0,
        "compatible": False,
        "name": None,
        "path": None,
        "ref": None,
        "label": None,
        "ranked": [],
    }
    if not entries:
        return blank
    ranked = []
    for e in entries:
        d = score_reference(query, e)
        d["label"] = e.get("label")
        d["ref"] = e
        ranked.append(d)
    ranked.sort(key=lambda d: d["score"], reverse=True)
    compat = [d for d in ranked if d.get("compatible")]
    chosen = compat[0] if compat else ranked[0]
    best = dict(chosen)
    best["ranked"] = ranked
    return best


def find_best_reference(query: dict, entries: list[dict]) -> dict:
    """Return the best GOOD view match. Empty library → unmatched."""
    blank = {
        "score": 0.0,
        "hist": 0.0,
        "shape": 0.0,
        "orb": 0.0,
        "embed": 0.0,
        "compatible": False,
        "name": None,
        "path": None,
        "ref": None,
        "ranked": [],
    }
    if not entries:
        return blank
    ranked = [score_reference(query, e) for e in entries]
    ranked.sort(key=lambda d: d["score"], reverse=True)
    chosen = next((d for d in ranked if d.get("compatible")), None)
    if chosen is None:
        blank["ranked"] = ranked
        return blank
    best = dict(chosen)
    best["ref"] = next(e for e in entries if e["name"] == best["name"])
    best["ranked"] = ranked
    return best
