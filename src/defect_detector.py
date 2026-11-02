"""Stages 6–8: local plastic anomaly vs the matched GOOD reference."""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from src.config import DEFAULT_PARAMS, STATUS_FAIL, STATUS_PASS, STATUS_REVIEW


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
    s1 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1 * mu1
    s2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1 * mu2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2) + 1e-8
    return np.clip(num / den, 0.0, 1.0)


def _lab(rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return cv2.split(lab)


def _yellow_metal_mask(rgb: np.ndarray, product: np.ndarray) -> np.ndarray:
    """
    Brass hex/threads that the insert HSV threshold missed on a live side view.
    Skip on end-on ports — dark gold in a bore is not a hex nut.
    """
    if product is None or int(cv2.countNonZero(product)) < 20:
        return np.zeros(rgb.shape[:2], np.uint8)
    ys, xs = np.where(product > 0)
    if len(xs) < 30:
        return np.zeros(rgb.shape[:2], np.uint8)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    eig = np.linalg.eigvalsh(np.cov(pts.T))
    eig = np.maximum(eig, 1e-6)
    ecc = float(np.sqrt(eig[1] / eig[0]))
    if ecc < 1.45:
        return np.zeros(rgb.shape[:2], np.uint8)
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    metal = (
        (product > 0)
        & (h >= 10)
        & (h <= 36)
        & (s >= 42)
        & (v >= 80)
    )
    mask = np.uint8(metal * 255)
    return cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )


def _intrinsic_plastic_marks(
    rgb: np.ndarray, plastic: np.ndarray, brass: np.ndarray, holes: np.ndarray | None = None
) -> List[dict]:
    """
    Marks that are visible on this part alone: dark spots or yellow/brown stain
    on cream. Independent of the reference photo (pose must not create them).
    """
    if int(cv2.countNonZero(plastic)) < 200:
        return []
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L, A, B = cv2.split(lab)
    S = hsv[:, :, 1]
    metal = _yellow_metal_mask(rgb, cv2.bitwise_or(plastic, brass))
    core = cv2.erode(plastic, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ignore = cv2.dilate(
        cv2.bitwise_or(brass, metal),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    core = cv2.bitwise_and(core, cv2.bitwise_not(ignore))
    if holes is not None:
        core = cv2.bitwise_and(core, cv2.bitwise_not(cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))))
    if int(cv2.countNonZero(core)) < 120:
        core = cv2.bitwise_and(plastic, cv2.bitwise_not(cv2.bitwise_or(brass, metal)))
    sel = core > 0
    med_l = float(np.median(L[sel]))
    med_a = float(np.median(A[sel]))
    med_b = float(np.median(B[sel]))
    da = A.astype(np.float32) - med_a
    db = B.astype(np.float32) - med_b
    chroma = np.sqrt(da * da + db * db)
    dark = (L.astype(np.float32) <= med_l - 26.0) & sel & (L > 35)
    stain = sel & (chroma >= 11.0) & (S.astype(np.float32) >= 12.0) & (L.astype(np.float32) <= med_l - 4.0) & (L > 70)
    mask = np.uint8((dark | stain) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dist = cv2.distanceTransform(plastic, cv2.DIST_L2, 5)
    h, w = L.shape[:2]
    blobs: List[dict] = []
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    core_n = max(int(np.count_nonzero(sel)), 1)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 100 or area > int(0.08 * core_n) or area > 1400:
            continue
        x, y, bw, bh = (int(stats[i, j]) for j in range(4))
        aspect = float(max(bw, bh) / max(min(bw, bh), 1))
        if aspect > 6.0:
            continue
        comp = labels == i
        cy, cx = int(cents[i][1]), int(cents[i][0])
        cy, cx = min(h - 1, max(0, cy)), min(w - 1, max(0, cx))
        if dist[cy, cx] < 5.0:
            continue
        dil = cv2.dilate(np.uint8(comp) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        ring = (dil > 0) & (~comp) & sel
        if int(np.count_nonzero(ring)) < 24:
            continue
        ring_L = float(np.mean(L[ring]))
        if ring_L < 150:
            continue
        contrast = ring_L - float(np.mean(L[comp]))
        ch = float(np.mean(chroma[comp]))
        if contrast < 32 and ch < 14:
            continue
        ys, xs = np.where(comp)
        blob_L = float(np.mean(L[comp]))
        kind = "color_spread" if ch >= 14 and contrast < 40 else "black_mark"
        if kind == "black_mark" and (area > 800 or blob_L > 155):
            continue
        if kind == "color_spread" and (area > 900 or blob_L > 195):
            continue
        # Dull brass reads as a compact yellow patch on cream. That is metal, not stain.
        mean_h = float(np.mean(hsv[:, :, 0][comp]))
        mean_s = float(np.mean(S[comp]))
        if kind == "color_spread" and 8.0 <= mean_h <= 38.0 and mean_s >= 38.0:
            continue
        blobs.append(
            {
                "bbox": [x, y, x + bw, y + bh],
                "area": area,
                "width": bw,
                "height": bh,
                "aspect_ratio": aspect,
                "mean_intensity": float(np.mean(L[comp])),
                "local_contrast": float(max(contrast, ch * 3.0)),
                "distance_from_boundary": float(dist[cy, cx]),
                "anomaly_score": float(min(1.2, contrast / 40.0 + ch / 18.0)),
                "cx": float(cx),
                "cy": float(cy),
                "contour": np.column_stack([xs, ys]),
                "label": "SUSPECTED DEFECT",
                "kind": kind,
            }
        )
    blobs.sort(key=lambda b: b["anomaly_score"] * np.sqrt(b["area"]), reverse=True)
    return blobs[:3]


def _marbling_blobs(
    rgb: np.ndarray,
    plastic: np.ndarray,
    brass: np.ndarray,
    holes: np.ndarray | None = None,
    view_kind: str = "side",
) -> List[dict]:
    """
    Grey/brown swirl on cream. Side-view T fittings only — end-on bores look
    brown from brass/shadow and must not count.
    """
    if (view_kind or "side") == "end_on":
        return []
    if int(cv2.countNonZero(plastic)) < 200:
        return []
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L, A, B = cv2.split(lab)
    S = hsv[:, :, 1]
    core = cv2.erode(plastic, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    metal = _yellow_metal_mask(rgb, cv2.bitwise_or(plastic, brass))
    core = cv2.bitwise_and(
        core,
        cv2.bitwise_not(
            cv2.dilate(
                cv2.bitwise_or(brass, metal),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            )
        ),
    )
    if holes is not None:
        core = cv2.bitwise_and(core, cv2.bitwise_not(cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))))
    sel = core > 0
    n = int(np.count_nonzero(sel))
    if n < 200:
        return []
    local_l = cv2.medianBlur(L, 25).astype(np.float32)
    streak = sel & (L.astype(np.float32) <= local_l - 16.0) & (L > 50) & (L < 205)
    brown = sel & (B.astype(np.float32) >= 140.0) & (S.astype(np.float32) >= 22.0) & (L.astype(np.float32) < 200)
    mask = np.uint8((streak | brown) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    marked = int(cv2.countNonZero(mask))
    frac = marked / float(n)
    if frac < 0.08 or marked < 320:
        return []
    blob = _blob(mask, "color_spread", score=float(min(1.2, 0.7 + frac)), contrast=max(28.0, frac * 200.0))
    if blob is None or blob["area"] < 250:
        return []
    return [blob]


def _blob(
    mask: np.ndarray,
    kind: str,
    score: float,
    contrast: float = 40.0,
    intensity: float = 0.0,
    dist: float = 8.0,
) -> dict | None:
    n, labels, stats, cents = cv2.connectedComponentsWithStats(np.uint8((mask > 0) * 255), 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 80:
        return None
    x, y, bw, bh = (int(stats[i, j]) for j in range(4))
    ys, xs = np.where(labels == i)
    return {
        "bbox": [x, y, x + bw, y + bh],
        "area": area,
        "width": bw,
        "height": bh,
        "aspect_ratio": float(max(bw, bh) / max(min(bw, bh), 1)),
        "mean_intensity": float(intensity),
        "local_contrast": float(contrast),
        "distance_from_boundary": float(dist),
        "anomaly_score": float(score),
        "cx": float(cents[i][0]),
        "cy": float(cents[i][1]),
        "contour": np.column_stack([xs, ys]),
        "label": "SUSPECTED DEFECT",
        "kind": kind,
    }


def _compact_puncture_blobs(
    holes: np.ndarray | None,
    product: np.ndarray,
    rgb: np.ndarray | None = None,
) -> List[dict]:
    """
    Star / black hole in the cream body. Flood-fill treats it as a port, so it
    never appears on the plastic mask. A real end-on bore is larger; missed
    brass threads are elongated.
    """
    if holes is None or product is None:
        return []
    prod_n = max(int(cv2.countNonZero(product)), 1)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(np.uint8(holes > 0), 8)
    if n <= 1:
        return []
    dist = cv2.distanceTransform(np.uint8(product > 0), cv2.DIST_L2, 5)
    L = S = None
    if rgb is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    blobs: List[dict] = []
    h, w = product.shape[:2]
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        frac = area / float(prod_n)
        if area < 280 or frac > 0.18:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = float(max(bw, bh) / max(min(bw, bh), 1))
        if aspect > 1.70:
            continue
        comp = np.uint8((labels == i) * 255)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = float(cv2.arcLength(c, True))
        ca = float(cv2.contourArea(c))
        circ = float(4.0 * np.pi * ca / (peri * peri + 1e-6))
        if circ < 0.30:
            continue
        cy, cx = int(cents[i][1]), int(cents[i][0])
        cy, cx = min(h - 1, max(0, cy)), min(w - 1, max(0, cx))
        if dist[cy, cx] < 3.5:
            continue
        ring = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        ring = (ring > 0) & (comp == 0) & (product > 0)
        if int(np.count_nonzero(ring)) < 24:
            continue
        if L is not None and S is not None:
            cream_ring = ring & (L >= 155) & (S <= 58)
            if int(np.count_nonzero(cream_ring)) < 20:
                continue
            if float(np.median(L[cream_ring])) < 155:
                continue
        blob = _blob(comp, "crushed_opening", score=1.18, contrast=58.0)
        if blob:
            blobs.append(blob)
    blobs.sort(key=lambda b: b["area"], reverse=True)
    return blobs[:1]


def _ink_blot_on_cream(
    rgb: np.ndarray,
    plastic: np.ndarray,
    brass: np.ndarray | None,
    holes: np.ndarray | None,
) -> List[dict]:
    """
    Black / grey blot on cream, including a puncture that was removed from the
    plastic mask. Brass threads and large end-on bores are skipped.
    """
    if int(cv2.countNonZero(plastic)) < 200:
        return []
    body = plastic.copy()
    if holes is not None:
        extra = np.zeros_like(plastic)
        ncc, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(holes > 0), 8)
        body_n = max(int(cv2.countNonZero(cv2.bitwise_or(plastic, holes))), 1)
        for i in range(1, ncc):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            aspect = float(max(bw, bh) / max(min(bw, bh), 1))
            if 180 <= area <= 9000 and area / float(body_n) <= 0.16 and aspect <= 1.85:
                extra[labels == i] = 255
        body = cv2.bitwise_or(body, extra)
    if brass is not None:
        body = cv2.bitwise_and(body, cv2.bitwise_not(brass))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    cream = plastic > 0
    if int(np.count_nonzero(cream)) < 150:
        return []
    med_l = float(np.median(L[cream]))
    dark = (body > 0) & (L.astype(np.float32) <= med_l - 48.0) & (L < 145)
    mask = np.uint8(dark * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    prod_n = max(int(cv2.countNonZero(body)), 1)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    blobs: List[dict] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 500 or area > int(0.16 * prod_n) or area > 9000:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = float(max(bw, bh) / max(min(bw, bh), 1))
        if aspect > 1.85:
            continue
        comp = labels == i
        if brass is not None:
            near = cv2.dilate(brass, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
            near_frac = float(np.count_nonzero(comp & (near > 0))) / float(max(area, 1))
        else:
            near_frac = 0.0
        hsv0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
        mean_s = float(np.mean(S[comp]))
        mean_h = float(np.mean(hsv0[comp]))
        if 8.0 <= mean_h <= 38.0 and mean_s >= 36.0:
            continue
        blob_l = float(np.mean(L[comp]))
        dil = cv2.dilate(np.uint8(comp) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        ring = (dil > 0) & (~comp) & cream
        if int(np.count_nonzero(ring)) < 20:
            continue
        ring_l = float(np.median(L[ring]))
        if ring_l < 155:
            continue
        contrast = ring_l - blob_l
        if contrast < 40:
            continue
        if near_frac > 0.20 and area < 1800 and contrast < 80:
            continue
        if area < 1500 and contrast < 70:
            continue
        ys, xs = np.where(comp)
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        blobs.append(
            {
                "bbox": [x, y, x + bw, y + bh],
                "area": area,
                "width": bw,
                "height": bh,
                "aspect_ratio": aspect,
                "mean_intensity": blob_l,
                "local_contrast": float(contrast),
                "distance_from_boundary": 8.0,
                "anomaly_score": float(min(1.2, contrast / 42.0)),
                "cx": float(cents[i][0]),
                "cy": float(cents[i][1]),
                "contour": np.column_stack([xs, ys]),
                "label": "SUSPECTED DEFECT",
                "kind": "black_mark",
            }
        )
    blobs.sort(key=lambda b: b["anomaly_score"] * np.sqrt(b["area"]), reverse=True)
    return blobs[:1]


def _crushed_opening_intrinsic(norm_kind: str, circ: float, hole_frac: float, ecc: float, product: np.ndarray) -> List[dict]:
    """
    End-on crushed port: the opening is no longer an enclosed hole and the
    silhouette is not round. Side-view T fittings are excluded (high ecc).
    """
    if norm_kind != "end_on" or hole_frac >= 0.08 or circ >= 0.42 or ecc >= 1.18:
        return []
    hull = np.zeros_like(product)
    cnts, _ = cv2.findContours(product, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cv2.drawContours(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, -1)
    missing = cv2.bitwise_and(hull, cv2.bitwise_not(product))
    missing = cv2.morphologyEx(missing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blob = _blob(missing, "crushed_opening", score=1.1, contrast=50.0)
    return [blob] if blob else []


def _side_crushed_end(
    product: np.ndarray,
    view_kind: str,
    ecc: float,
    rgb: np.ndarray,
    brass: np.ndarray | None,
) -> List[dict]:
    """
    Side-view T/elbow: one arm is pinched/crushed so its width collapses,
    while the other end stays a normal opening.

    A brass hex adapter tapers at BOTH ends and is ignored.
    """
    if (view_kind or "side") == "end_on" or ecc < 1.22:
        return []
    if int(cv2.countNonZero(product)) < 400:
        return []
    metal = _yellow_metal_mask(rgb, product)
    if brass is not None:
        metal = cv2.bitwise_or(metal, brass)
    prod_n = max(int(cv2.countNonZero(product)), 1)
    if int(cv2.countNonZero(metal)) / prod_n > 0.08:
        return []

    ys, xs = np.where(product > 0)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(0)
    _eig, vec = np.linalg.eigh(np.cov(pts.T))
    axis = vec[:, 1]
    perp = vec[:, 0]
    proj = (pts - mean) @ axis
    t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    n_bins = 20
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    widths = np.full(n_bins, np.nan, np.float32)
    for i in range(n_bins):
        m = (t >= bins[i]) & (t < bins[i + 1])
        if int(m.sum()) < 8:
            continue
        widths[i] = float(np.ptp((pts[m] - mean) @ perp))

    valid = ~np.isnan(widths)
    if int(valid.sum()) < 12:
        return []
    mid = float(np.nanmedian(widths[8:12])) if np.any(~np.isnan(widths[8:12])) else float(np.nanmedian(widths))
    if mid < 40:
        return []
    cut = 0.55 * mid

    def _run(from_end: bool) -> int:
        seq = widths[::-1] if from_end else widths
        n = 0
        for w in seq:
            if np.isnan(w):
                continue
            if w < cut:
                n += 1
            else:
                break
        return n

    n0, n1 = _run(False), _run(True)
    w0 = float(np.nanmedian(widths[:4]))
    w1 = float(np.nanmedian(widths[-4:]))
    if min(w0, w1) < 1:
        return []
    ratio = min(w0, w1) / max(w0, w1)
    if max(n0, n1) < 3 or abs(n1 - n0) < 2 or ratio > 0.62:
        return []

    # Mask the crushed end band for the overlay.
    crushed_high = n1 > n0
    t0, t1 = (0.80, 1.01) if crushed_high else (-0.01, 0.20)
    band = np.zeros_like(product)
    sel_pts = pts[(t >= t0) & (t <= t1)]
    if len(sel_pts) < 30:
        return []
    band[sel_pts[:, 1].astype(int), sel_pts[:, 0].astype(int)] = 255
    band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    band = cv2.bitwise_and(band, product)
    blob = _blob(band, "crushed_opening", score=1.22, contrast=60.0)
    return [blob] if blob else []


def _color_spread_vs_ref(q_rgb: np.ndarray, q_plastic: np.ndarray, q_brass: np.ndarray, ref: dict) -> List[dict]:
    """Whole-body yellow/brown marbling versus a clean cream GOOD of the same view."""
    if ref is None:
        return []
    q_prod = max(int(cv2.countNonZero(q_plastic)) + int(cv2.countNonZero(q_brass)), 1)
    q_brass_frac = float(cv2.countNonZero(q_brass)) / q_prod
    if q_brass_frac > 0.12 or float(ref.get("brass_frac") or 0) > 0.12:
        return []
    q_hsv = cv2.cvtColor(q_rgb, cv2.COLOR_RGB2HSV)
    r_hsv = cv2.cvtColor(ref["rgb"], cv2.COLOR_RGB2HSV)
    q_metal = _yellow_metal_mask(q_rgb, cv2.bitwise_or(q_plastic, q_brass))
    q_sel = (q_plastic > 0) & (q_brass == 0) & (q_metal == 0)
    r_brass = ref.get("brass_mask", np.zeros_like(q_plastic))
    r_metal = _yellow_metal_mask(ref["rgb"], cv2.bitwise_or(ref["plastic_mask"], r_brass))
    r_sel = (ref["plastic_mask"] > 0) & (r_brass == 0) & (r_metal == 0)
    if int(np.count_nonzero(q_sel)) < 200 or int(np.count_nonzero(r_sel)) < 200:
        return []
    qS = q_hsv[:, :, 1]
    rS = r_hsv[:, :, 1]
    dS = float(np.median(qS[q_sel])) - float(np.median(rS[r_sel]))
    if dS < 16.0:
        return []
    cut = float(np.median(rS[r_sel])) + 12.0
    stain = np.uint8((q_sel & (qS.astype(np.float32) >= cut)) * 255)
    stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    stain = cv2.morphologyEx(stain, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    blob = _blob(stain, "color_spread", score=min(1.2, dS / 22.0), contrast=dS)
    return [blob] if blob and blob["area"] >= 400 else []


def _rim_dent_blobs(product: np.ndarray, circ: float, hole_frac: float, ecc: float) -> List[dict]:
    """Localized crushed rim on a round end-on port (not a T crotch or elbow arm)."""
    if circ < 0.50 or hole_frac < 0.18 or ecc > 1.40:
        return []
    cnts, _ = cv2.findContours(product, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    if radius < 40:
        return []
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    order = np.argsort(ang)
    ang, dist = ang[order], dist[order]
    bins = np.linspace(-np.pi, np.pi, 72)
    prof = []
    for i in range(len(bins) - 1):
        m = (ang >= bins[i]) & (ang < bins[i + 1])
        prof.append(float(np.median(dist[m])) if np.any(m) else np.nan)
    prof = np.array(prof, np.float32)
    med = float(np.nanmedian(prof))
    prof = np.where(np.isnan(prof), med, prof)
    inward = (med - prof) / max(med, 1.0)
    deep = inward > 0.16
    best_run, best_depth, best_i, run, depth, start = 0, 0.0, 0, 0, 0.0, 0
    wrap = np.concatenate([deep, deep])
    win = np.concatenate([inward, inward])
    run_start = 0
    for i, v in enumerate(wrap):
        if v:
            if run == 0:
                run_start = i
            run += 1
            depth = max(depth, float(win[i]))
            if run > best_run:
                best_run, best_depth, best_i = run, depth, run_start
        else:
            run, depth = 0, 0.0
    best_run = min(best_run, 71)
    deg = best_run * (360.0 / 71.0)
    rest = inward[~deep] if np.any(~deep) else inward
    if float(np.std(rest)) > 0.09:
        return []
    if best_depth < 0.24 or deg < 18 or deg > 70:
        return []
    mask = np.zeros(product.shape[:2], np.uint8)
    i0 = best_i % 71
    for k in range(best_run):
        t0 = bins[(i0 + k) % 71]
        t1 = bins[(i0 + k + 1) % 71]
        sel = (ang >= min(t0, t1)) & (ang < max(t0, t1)) if t1 > t0 else (ang >= t0) | (ang < t1)
        if np.any(sel):
            xy = pts[order][sel].astype(np.int32)
            mask[np.clip(xy[:, 1], 0, mask.shape[0] - 1), np.clip(xy[:, 0], 0, mask.shape[1] - 1)] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    mask = cv2.bitwise_and(mask, product)
    blob = _blob(mask, "crushed_opening", score=1.15, contrast=best_depth * 100.0)
    return [blob] if blob else []


def _silhouette_xor_blobs(aligned_prod: np.ndarray, ref_prod: np.ndarray) -> List[dict]:
    """Localized missing/extra material after alignment — broken opening."""
    a = cv2.morphologyEx(aligned_prod, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    b = cv2.morphologyEx(ref_prod, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    missing = cv2.bitwise_and(b, cv2.bitwise_not(a))
    extra = cv2.bitwise_and(a, cv2.bitwise_not(b))
    xor = cv2.bitwise_or(missing, extra)
    xor = cv2.morphologyEx(xor, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    h, w = a.shape[:2]
    union_n = max(int(cv2.countNonZero(cv2.bitwise_or(a, b))), 1)
    blobs: List[dict] = []
    n, labels, stats, cents = cv2.connectedComponentsWithStats(xor, 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 280 or area > int(0.10 * union_n):
            continue
        x, y, bw, bh = (int(stats[i, j]) for j in range(4))
        comp = labels == i
        ys, xs = np.where(comp)
        blobs.append(
            {
                "bbox": [x, y, x + bw, y + bh],
                "area": area,
                "width": bw,
                "height": bh,
                "aspect_ratio": float(max(bw, bh) / max(min(bw, bh), 1)),
                "mean_intensity": 0.0,
                "local_contrast": 40.0,
                "distance_from_boundary": 6.0,
                "anomaly_score": float(min(1.1, area / 800.0)),
                "cx": float(cents[i][0]),
                "cy": float(cents[i][1]),
                "contour": np.column_stack([xs, ys]),
                "label": "SUSPECTED DEFECT",
                "kind": "broken_opening",
            }
        )
    blobs.sort(key=lambda b: b["area"], reverse=True)
    return blobs[:2]


def _illum_shift(q_ch: np.ndarray, r_ch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift query channel so its median on mask matches the reference. Global lighting only."""
    sel = mask > 0
    q = q_ch.astype(np.float32)
    if int(np.count_nonzero(sel)) < 80:
        return q
    qmed = float(np.median(q[sel]))
    rmed = float(np.median(r_ch.astype(np.float32)[sel]))
    return np.clip(q + (rmed - qmed), 0.0, 255.0)


def detect_local_anomalies(aligned: dict, ref: dict | None, params: Dict | None = None) -> dict:
    """
    Compare aligned query to the best GOOD reference on plastic only.
    Brass, holes, borders, glare, and whole-part view mismatch are ignored.
    Intrinsic dark/stain marks on this crop are included even if the
    reference view is imperfect.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    q_rgb = aligned["rgb"]
    h, w = q_rgb.shape[:2]
    qL, qA, qB = _lab(q_rgb)
    blobs: List[dict] = []
    anomaly = np.zeros((h, w), np.float32)
    cand = np.zeros((h, w), np.uint8)
    interior = np.zeros((h, w), np.uint8)
    mean_ssim = 1.0
    search_n = 0
    hot_total = 1
    geom_ok = int(aligned.get("inliers") or 0) >= 10

    if ref is not None:
        r_rgb = ref["rgb"]
        rL, rA, rB = _lab(r_rgb)
        plastic = cv2.bitwise_and(aligned["plastic_mask"], ref["plastic_mask"])
        brass = cv2.bitwise_or(aligned["brass_mask"], ref["brass_mask"])
        metal = _yellow_metal_mask(q_rgb, aligned["product_mask"])
        brass = cv2.bitwise_or(brass, metal)
        brass = cv2.dilate(brass, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        holes = cv2.bitwise_or(
            aligned.get("holes_mask", np.zeros((h, w), np.uint8)),
            ref.get("holes_mask", np.zeros((h, w), np.uint8)),
        )
        holes = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        erode_px = max(2, int(p["boundary_erode"]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        interior = cv2.erode(plastic, k, iterations=1)
        interior = cv2.bitwise_and(interior, cv2.bitwise_not(brass))
        interior = cv2.bitwise_and(interior, cv2.bitwise_not(holes))
        qL_m = _illum_shift(qL, rL, interior)
        qA_m = _illum_shift(qA, rA, interior)
        qB_m = _illum_shift(qB, rB, interior)

        ssim = _ssim_map(qL_m, rL)
        dL = np.abs(qL_m - rL.astype(np.float32)) / 255.0
        dC = (
            np.sqrt((qA_m - rA.astype(np.float32)) ** 2 + (qB_m - rB.astype(np.float32)) ** 2)
            / 40.0
        )
        anomaly = (
            float(p["luma_weight"]) * dL
            + float(p["chroma_weight"]) * np.clip(dC, 0, 1.5)
            + float(p["ssim_weight"]) * (1.0 - ssim)
        )
        anomaly[qL > 242] = 0.0
        anomaly[interior == 0] = 0.0
        search_n = max(int(cv2.countNonZero(interior)), 1)
        mean_ssim = float(np.mean(ssim[interior > 0])) if search_n > 20 else 1.0

        cut = float(p["anomaly_cut"])
        if search_n > 80:
            p90 = float(np.percentile(anomaly[interior > 0], 90))
            cut = max(cut, p90 * 0.85)
        cand = np.uint8((anomaly >= cut) & (interior > 0)) * 255
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        min_area = int(p["min_defect_area"])
        max_area = int(search_n * float(p["max_defect_frac"]))
        dist_map = cv2.distanceTransform(np.uint8((plastic > 0) * 255), cv2.DIST_L2, 5)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
        hot_total = max(int(cv2.countNonZero(cand)), 1)
        med_own = (
            float(np.median(qL[aligned["plastic_mask"] > 0]))
            if int(cv2.countNonZero(aligned["plastic_mask"])) > 50
            else 200.0
        )
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = (int(stats[i, j]) for j in range(4))
            comp = labels == i
            aspect = float(max(bw, bh) / max(min(bw, bh), 1))
            if aspect > 7.5:
                continue
            cy, cx = int(cents[i][1]), int(cents[i][0])
            cy = min(h - 1, max(0, cy))
            cx = min(w - 1, max(0, cx))
            if dist_map[cy, cx] < 4.0:
                continue
            score = float(np.mean(anomaly[comp]))
            contrast_ref = float(np.mean(np.abs(qL_m[comp] - rL[comp].astype(np.float32))))
            contrast_own = med_own - float(np.mean(qL[comp]))
            if contrast_own < 10 and contrast_ref < 22:
                continue
            # Residual next to missed brass (hex / threads) is not a plastic stain.
            if int(np.count_nonzero(comp & (brass > 0))) > 0.12 * area:
                continue
            ys, xs = np.where(comp)
            blobs.append(
                {
                    "bbox": [x, y, x + bw, y + bh],
                    "area": area,
                    "width": bw,
                    "height": bh,
                    "aspect_ratio": aspect,
                    "mean_intensity": float(np.mean(qL[comp])),
                    "local_contrast": float(max(contrast_ref, contrast_own)),
                    "own_contrast": float(contrast_own),
                    "distance_from_boundary": float(dist_map[cy, cx]),
                    "anomaly_score": score,
                    "cx": float(cx),
                    "cy": float(cy),
                    "contour": np.column_stack([xs, ys]),
                    "label": "SUSPECTED DEFECT",
                    "kind": "local_anomaly",
                }
            )
        blobs.extend(_color_spread_vs_ref(q_rgb, aligned["plastic_mask"], aligned["brass_mask"], ref))
        if geom_ok:
            blobs.extend(_silhouette_xor_blobs(aligned["product_mask"], ref["product_mask"]))

    blobs.extend(
        _intrinsic_plastic_marks(
            q_rgb,
            aligned["plastic_mask"],
            aligned["brass_mask"],
            aligned.get("holes_mask"),
        )
    )
    blobs.extend(
        _marbling_blobs(
            q_rgb,
            aligned["plastic_mask"],
            aligned["brass_mask"],
            aligned.get("holes_mask"),
            str(aligned.get("view_kind") or "side"),
        )
    )
    blobs.extend(
        _compact_puncture_blobs(
            aligned.get("holes_mask"),
            aligned["product_mask"],
            q_rgb,
        )
    )
    blobs.extend(
        _ink_blot_on_cream(
            q_rgb,
            aligned["plastic_mask"],
            aligned.get("brass_mask"),
            aligned.get("holes_mask"),
        )
    )
    blobs.extend(
        _crushed_opening_intrinsic(
            str(aligned.get("view_kind") or "side"),
            float(aligned.get("circularity") or 0.0),
            float(aligned.get("hole_frac") or 0.0),
            float(aligned.get("eccentricity") or 1.0),
            aligned["product_mask"],
        )
    )
    blobs.extend(
        _side_crushed_end(
            aligned["product_mask"],
            str(aligned.get("view_kind") or "side"),
            float(aligned.get("eccentricity") or 1.0),
            q_rgb,
            aligned.get("brass_mask"),
        )
    )
    blobs.extend(
        _rim_dent_blobs(
            aligned["product_mask"],
            float(aligned.get("circularity") or 0.0),
            float(aligned.get("hole_frac") or 0.0),
            float(aligned.get("eccentricity") or 1.0),
        )
    )
    def _rank(b):
        intrinsic = 2.0 if b.get("kind") in ("black_mark", "color_spread", "crushed_opening") else 1.0
        return intrinsic * b["anomaly_score"] * np.sqrt(max(b["area"], 1))

    blobs.sort(key=_rank, reverse=True)
    blobs = blobs[:3]
    best = blobs[0] if blobs else None
    local_score = 0.0
    concentrated = 0.0
    if best is not None:
        local_score = float(
            min(
                1.6,
                best["anomaly_score"] * (np.log1p(best["area"]) / 5.2)
                + 0.012 * float(best.get("local_contrast") or 0.0),
            )
        )
        concentrated = float(best["area"] / hot_total)
    return {
        "anomaly_map": anomaly.astype(np.float32),
        "candidate_mask": cand,
        "interior_mask": interior,
        "blobs": blobs,
        "best": best,
        "local_score": float(local_score),
        "mean_ssim": mean_ssim,
        "ref_diff": float(max(0.0, 1.0 - mean_ssim)),
        "concentrated": float(concentrated),
        "search_n": search_n,
        "align_ok": bool(geom_ok),
    }


def decide(
    local_score: float,
    match_score: float,
    ref_diff: float,
    params: Dict | None = None,
    align_ok: bool = False,
    best_kind: str | None = None,
    best_area: int = 0,
    best_contrast: float = 0.0,
    own_contrast: float = 0.0,
    view_kind: str = "side",
) -> tuple[str, str]:
    """
    Simple station rule using only dataset/good as the reference:
    - crushed / missing opening, colour spread, or a large dark stain → DEFECTIVE
    - glare, brass, mold text, and small specks are ignored → GOOD
    Live camera: ignore pose/lighting residuals unless the blob is dark on THIS part.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    fail_t = float(p["fail_local"])
    live = bool(p.get("live_mode"))
    real_defect = False
    why = "GOOD. Compared with the GOOD photos in dataset/good."
    if best_kind == "crushed_opening" and local_score >= fail_t:
        real_defect, why = True, "DEFECTIVE. Opening is crushed or incomplete."
    elif best_kind == "color_spread" and local_score >= fail_t and best_area >= 250:
        real_defect, why = True, "DEFECTIVE. Yellow/brown colour spread on the plastic."
    elif best_kind == "black_mark" and local_score >= fail_t and best_area >= 100 and best_contrast >= 32:
        real_defect, why = True, "DEFECTIVE. Dark stain on the cream plastic."
    elif best_kind == "broken_opening" and align_ok and local_score >= fail_t:
        real_defect, why = True, "DEFECTIVE. Opening does not match the GOOD photos."
    elif best_kind == "local_anomaly" and best_area >= 700:
        if live or view_kind == "side":
            # Live lighting and brass-hex adapters create residuals vs the GOOD photos.
            # Fail only for a dark mark on this part, or a very strong residual vs reference.
            if own_contrast >= 48 or (not live and best_contrast >= 55):
                real_defect, why = True, "DEFECTIVE. Dark damage on the cream plastic."
        elif best_contrast >= 50 or (
            best_area >= 900 and own_contrast >= 15 and best_contrast >= 32
        ):
            real_defect, why = True, "DEFECTIVE. Clear damage versus the GOOD photos."
    if real_defect:
        return STATUS_FAIL, why
    return STATUS_PASS, why
