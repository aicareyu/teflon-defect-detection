"""
Class activation heatmap (CAM / Grad-CAM equivalent).

This highlights which spatial region of the *product image* most influenced
the classifier. It is NOT a pixel-accurate defect segmentation and must not
be presented as one.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.model import ResNet18Encoder, linear_weights_for_cam


def _normalise_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0.0)
    vmin, vmax = float(cam.min()), float(cam.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    cam = (cam - vmin) / (vmax - vmin)
    return cam.astype(np.float32)


def cam_from_feature_map(
    feature_map: np.ndarray,
    class_weights: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """
    Original CAM: weighted sum of last conv channels.

    feature_map: (C, H, W)  — typically (512, 7, 7) for ResNet-18
    class_weights: (n_classes, C) in embedding space
    """
    w = class_weights[class_index]
    cam = np.tensordot(w, feature_map, axes=(0, 0))
    return _normalise_cam(cam)


def overlay_heatmap(
    rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize CAM to the image, return (colour heatmap RGB, overlay RGB)."""
    h, w = rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_u8 = np.uint8(np.clip(cam_resized * 255.0, 0, 255))
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 1.0 - alpha, heat_rgb, alpha, 0)
    return heat_rgb, overlay


@torch.no_grad()
def sklearn_cam(
    encoder: ResNet18Encoder,
    tensor: torch.Tensor,
    pipeline,
    class_index: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    weights = linear_weights_for_cam(pipeline)
    if weights is None:
        return None
    encoder.eval()
    encoder.to(device)
    _, fmap = encoder(tensor.to(device))
    fmap_np = fmap[0].cpu().numpy()
    return cam_from_feature_map(fmap_np, weights, class_index)


def gradcam_pytorch_head(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    class_index: int,
    device: torch.device,
) -> np.ndarray:
    """
    Grad-CAM on ResNet-18 layer4 for the fine-tuned head path.
    """
    model.eval()
    model.to(device)
    x = tensor.to(device)
    activations = {}
    gradients = {}

    def fwd_hook(_m, _i, output):
        activations["v"] = output

    def bwd_hook(_m, _gin, gout):
        gradients["v"] = gout[0]

    layer = model.net.layer4
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(x)
        score = logits[0, class_index]
        model.zero_grad(set_to_none=True)
        score.backward()
        acts = activations["v"]
        grads = gradients["v"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam[0, 0].detach().cpu().numpy()
        return _normalise_cam(cam)
    finally:
        h1.remove()
        h2.remove()
