"""Stage 6: align the query crop to the chosen GOOD reference."""

from __future__ import annotations

import cv2
import numpy as np


def _homography(q_kps, q_desc, r_kps, r_desc):
    if q_desc is None or r_desc is None or len(q_kps) < 8 or len(r_kps) < 8:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(q_desc, r_desc, k=2)
    good = []
    for item in knn:
        if len(item) < 2:
            continue
        a, b = item
        if a.distance < 0.75 * b.distance:
            good.append(a)
    if len(good) < 8:
        return None, len(good)
    src = np.float32([q_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([r_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < 6:
        return None, inliers
    return H, inliers


def _ecc_warp(q_gray: np.ndarray, r_gray: np.ndarray, q_mask: np.ndarray):
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        qn = cv2.equalizeHist(q_gray)
        rn = cv2.equalizeHist(r_gray)
        corr = cv2.findTransformECC(
            rn,
            qn,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
            q_mask,
            1,
        )
        if isinstance(corr, tuple):
            corr = corr[0]
        return warp, float(corr)
    except cv2.error:
        return None, 0.0


def _warp_rgb(img: np.ndarray, M, size, is_homography: bool) -> np.ndarray:
    h, w = size
    if is_homography:
        return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(18, 18, 18))
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(18, 18, 18))


def _warp_mask(img: np.ndarray, M, size, is_homography: bool) -> np.ndarray:
    h, w = size
    flags = cv2.INTER_NEAREST
    if is_homography:
        return cv2.warpPerspective(img, M, (w, h), flags=flags, borderValue=0)
    return cv2.warpAffine(img, M, (w, h), flags=flags, borderValue=0)


def align_to_reference(query: dict, ref: dict) -> dict:
    """
    Warp the query onto the reference canvas.
    Homography when ORB supports it; ECC affine otherwise; identity last.
    """
    h, w = ref["rgb"].shape[:2]
    size = (h, w)
    method = "resize"
    inliers = 0
    H, n = _homography(query.get("kps"), query.get("desc"), ref.get("kps"), ref.get("desc"))
    ecc_corr = 0.0
    if H is not None:
        method = "homography"
        inliers = n
        M, homo = H, True
    else:
        # Do not ECC-warp incompatible poses. That stretch is not a defect.
        method = "identity"
        M, homo = np.eye(3, dtype=np.float64), True
        ecc_corr = 0.0

    aligned_rgb = _warp_rgb(query["rgb"], M, size, homo)
    aligned = {
        "rgb": aligned_rgb,
        "gray": cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY),
        "product_mask": _warp_mask(query["product_mask"], M, size, homo),
        "plastic_mask": _warp_mask(query["plastic_mask"], M, size, homo),
        "brass_mask": _warp_mask(query["brass_mask"], M, size, homo),
        "holes_mask": _warp_mask(query.get("holes_mask", np.zeros_like(query["product_mask"])), M, size, homo),
        "method": method,
        "inliers": int(inliers),
        "ecc_corr": float(ecc_corr),
        "matrix": M,
    }
    return aligned
