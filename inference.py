"""
Image-level inference for the plastic fitting defect classifier.

Usage:
    python inference.py path/to/image.jpg

API:
    predict_image(image_path) -> dict with prediction, confidence, probabilities
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    DISPLAY_NAMES,
    MODEL_PATH,
    USE_PRODUCT_CROP,
)
from src.gradcam import (  # noqa: E402
    gradcam_pytorch_head,
    overlay_heatmap,
    sklearn_cam,
)
from src.inspection import decide_status, format_result_block  # noqa: E402
from src.model import (  # noqa: E402
    ResNet18Encoder,
    TransferHeadClassifier,
    get_device,
    load_checkpoint,
    sklearn_predict_proba,
)
from src.preprocess import imagenet_tensor, prepare_from_path, prepare_rgb  # noqa: E402

_CACHE = {"ckpt": None, "encoder": None, "finetune": None, "device": None}


def clear_runtime_cache() -> None:
    """Call after retraining so the next inspect loads the new weights."""
    _CACHE.update({"ckpt": None, "encoder": None, "finetune": None, "device": None})


def _load_runtime(model_path: Optional[Path] = None):
    path = Path(model_path) if model_path else MODEL_PATH
    if _CACHE["ckpt"] is not None and path == MODEL_PATH:
        return _CACHE["ckpt"], _CACHE["encoder"], _CACHE["finetune"], _CACHE["device"]

    ckpt = load_checkpoint(path)
    device = get_device()
    encoder = ResNet18Encoder()
    encoder.to(device)
    encoder.eval()
    finetune = None
    if ckpt.get("approach") == "finetune_resnet18":
        finetune = TransferHeadClassifier()
        finetune.load_state_dict(ckpt["finetune_state_dict"])
        finetune.to(device)
        finetune.eval()
    if path == MODEL_PATH:
        _CACHE.update(
            {"ckpt": ckpt, "encoder": encoder, "finetune": finetune, "device": device}
        )
    return ckpt, encoder, finetune, device


def _probabilities(ckpt: dict, encoder, finetune, tensor, device) -> np.ndarray:
    if ckpt.get("approach") == "finetune_resnet18" and finetune is not None:
        with torch.no_grad():
            logits = finetune(tensor.to(device))
            proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return proba
    pipe = ckpt["sklearn_pipeline"]
    with torch.no_grad():
        emb, _ = encoder(tensor.to(device))
    X = emb.cpu().numpy()
    return sklearn_predict_proba(pipe, X)[0]


def predict_image(
    image_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    use_crop: Optional[bool] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Run Good vs Not-Good classification on a single image.

    Returns
    -------
    dict
        prediction : "GOOD" | "NOT GOOD"
        confidence : float
        good_probability : float
        not_good_probability : float
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path.as_posix()}")

    ckpt, encoder, finetune, device = _load_runtime(
        Path(model_path) if model_path else None
    )
    if use_crop is None:
        use_crop = bool(ckpt.get("use_crop", USE_PRODUCT_CROP))

    original, display, net, crop_info = prepare_from_path(image_path, use_crop=use_crop)
    tensor = imagenet_tensor(net)
    proba = _probabilities(ckpt, encoder, finetune, tensor, device)
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    prediction = DISPLAY_NAMES[pred_idx]
    status = decide_status(pred_idx, confidence, threshold)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "good_probability": float(proba[0]),
        "not_good_probability": float(proba[1]),
        "predicted_index": pred_idx,
        "status": status,
        "threshold": float(threshold),
        "crop_info": crop_info,
        "filename": image_path.name,
        "original_rgb": original,
        "display_rgb": display,
        "network_rgb": net,
    }


def explain_image(
    image_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    use_crop: Optional[bool] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """predict_image plus CAM heatmap and overlay on the product crop."""
    result = predict_image(
        image_path, model_path=model_path, use_crop=use_crop, threshold=threshold
    )
    ckpt, encoder, finetune, device = _load_runtime(
        Path(model_path) if model_path else None
    )
    tensor = imagenet_tensor(result["network_rgb"])
    pred_idx = result["predicted_index"]
    cam = None
    method = "none"

    if ckpt.get("approach") == "finetune_resnet18" and finetune is not None:
        cam = gradcam_pytorch_head(finetune, tensor, pred_idx, device)
        method = "gradcam_layer4"
    else:
        cam = sklearn_cam(
            encoder, tensor, ckpt["sklearn_pipeline"], pred_idx, device
        )
        method = "cam_linear_gap" if cam is not None else "unavailable"

    vis_src = result["display_rgb"]
    if cam is None:
        heat = np.zeros_like(vis_src)
        overlay = vis_src.copy()
    else:
        heat, overlay = overlay_heatmap(vis_src, cam)

    result.update(
        {
            "attention_method": method,
            "heatmap_rgb": heat,
            "overlay_rgb": overlay,
            "cam": cam,
            "attention_note": (
                "Model attention / suspected defect region. "
                "This heatmap is not a precise defect segmentation."
            ),
            "result_text": format_result_block(
                result["status"], result["prediction"], result["confidence"]
            ),
        }
    )
    return result


def classify_from_rgb(
    rgb: np.ndarray,
    filename: str = "camera.png",
    model_path: Optional[Union[str, Path]] = None,
    use_crop: Optional[bool] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    with_visuals: bool = False,
) -> dict:
    from src.gallery import classify_rgb

    crop = True if use_crop is None else bool(use_crop)
    return classify_rgb(rgb, filename=filename, threshold=threshold, use_crop=crop)


def predict_from_rgb(
    rgb: np.ndarray,
    filename: str = "uploaded.png",
    model_path: Optional[Union[str, Path]] = None,
    use_crop: Optional[bool] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Same pipeline as explain_image, from an in-memory RGB array (upload/camera)."""
    return classify_from_rgb(
        rgb,
        filename=filename,
        model_path=model_path,
        use_crop=use_crop,
        threshold=threshold,
        with_visuals=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one fitting image.")
    parser.add_argument("image", help="Path to a JPEG/PNG image")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    parser.add_argument("--no-crop", action="store_true")
    args = parser.parse_args()
    try:
        out = predict_image(
            args.image,
            use_crop=False if args.no_crop else None,
            threshold=args.threshold,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    printable = {
        k: v
        for k, v in out.items()
        if k
        not in {
            "original_rgb",
            "display_rgb",
            "network_rgb",
            "crop_info",
        }
    }
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
