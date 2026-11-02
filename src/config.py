"""Proof-of-concept inspection parameters. Not a production lock."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "care_yu_logo_light.png"
LOGO_MARK_PATH = ASSETS_DIR / "care_yu_mark_light.png"

DATASET_DIR = PROJECT_ROOT / "dataset"
GOOD_DIR = DATASET_DIR / "good"
NOT_GOOD_DIR = DATASET_DIR / "not_good"
BAD_DIR = NOT_GOOD_DIR
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
INSPECTION_LOG_CSV = LOGS_DIR / "inspection_log.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_REVIEW = "REVIEW"

CAMERA_DEFAULT_INDEX = 2
CAMERA_PROBE_MAX_INDEX = 6
LIVE_PREVIEW_MAX_WIDTH = 960
LIVE_PRESENT_AREA_MIN = 0.048
LIVE_EMPTY_AREA_MAX = 0.04
LIVE_EMPTY_FRAMES = 4
LIVE_MOTION_STABLE = 4.5
LIVE_STABLE_FRAMES = 3
LIVE_MIN_BBOX_WIDTH_FRAC = 0.14
LIVE_MIN_BBOX_HEIGHT_FRAC = 0.12

CANVAS_SIZE = 400
ORB_FEATURES = 400

# Compatibility leftovers for unused train/gallery scripts.
CLASS_TO_IDX = {"good": 0, "not_good": 1}
IDX_TO_CLASS = {0: "good", 1: "not_good"}
DISPLAY_NAMES = {"good": "GOOD", "not_good": "NOT GOOD"}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LETTERBOX_FILL_RGB = (18, 18, 18)
MODEL_PATH = MODELS_DIR / "resnet18_head.pt"
GALLERY_PATH = MODELS_DIR / "gallery.npz"
BACKBONE_NAME = "resnet18"

DEFAULT_PARAMS = {
    "match_min": 0.30,
    "fail_local": 0.62,
    "review_local": 0.38,
    "min_defect_area": 90,
    "max_defect_frac": 0.18,
    "boundary_erode": 6,
    "anomaly_cut": 0.20,
    "ssim_weight": 0.30,
    "luma_weight": 0.45,
    "chroma_weight": 0.25,
}

SMALL_DATA_WARNING = (
    "Proof-of-concept inspection system. Current sample count is too small "
    "for production validation. Production deployment requires controlled "
    "imaging conditions and a significantly larger validated dataset."
)


def list_dataset_images() -> dict:
    goods = (
        sorted(p for p in GOOD_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if GOOD_DIR.exists()
        else []
    )
    ngs = (
        sorted(p for p in NOT_GOOD_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if NOT_GOOD_DIR.exists()
        else []
    )
    return {"good": goods, "not_good": ngs, "bad": ngs}
