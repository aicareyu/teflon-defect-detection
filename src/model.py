"""
Backbone + classifiers for tiny-dataset Good vs Not-Good inspection.

Preferred path for ~15 images
-----------------------------
Pretrained ResNet-18 (ImageNet) as a frozen feature extractor
    -> StandardScaler
    -> LogisticRegression or Linear SVM

Optional comparison path
------------------------
Frozen ResNet-18 + trainable linear head (transfer learning).

A deep net is never trained from scratch here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from torchvision.models import ResNet18_Weights, resnet18

from src.config import BACKBONE_NAME, IMAGE_SIZE, MODEL_PATH


class ResNet18Encoder(nn.Module):
    """Frozen ImageNet ResNet-18. Returns GAP embedding and last conv map."""

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        # children: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.features = nn.Sequential(*list(net.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap = self.features(x)
        emb = self.pool(fmap).flatten(1)
        return emb, fmap


class TransferHeadClassifier(nn.Module):
    """Frozen backbone + trainable 2-class head (comparison path)."""

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        for p in net.parameters():
            p.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, 2)
        for p in net.fc.parameters():
            p.requires_grad = True
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def embeddings_and_map(self, x: torch.Tensor):
        fmap = nn.Sequential(*list(self.net.children())[:-2])(x)
        emb = self.net.avgpool(fmap).flatten(1)
        return emb, fmap


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract_embeddings(
    encoder: ResNet18Encoder,
    tensors: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    tensors: N x 3 x H x W
    Returns embeddings (N, 512) and per-image feature maps (512, h, w).
    """
    encoder.eval()
    encoder.to(device)
    embs = []
    maps = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i : i + batch_size].to(device)
        emb, fmap = encoder(batch)
        embs.append(emb.cpu().numpy())
        for j in range(fmap.size(0)):
            maps.append(fmap[j].cpu().numpy())
    return np.concatenate(embs, axis=0), maps


def make_sklearn_candidates(random_state: int = 42) -> List[Tuple[str, Pipeline]]:
    """Linear models are preferred; RBF SVM is included as a reference only."""
    candidates = [
        (
            "logreg",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            C=1.0,
                            solver="lbfgs",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "linear_svm",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LinearSVC(
                            class_weight="balanced",
                            C=1.0,
                            dual=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "rbf_svm",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            C=1.0,
                            gamma="scale",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    ]
    return candidates


def sklearn_predict_proba(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Uniform probability interface (LinearSVC has no predict_proba)."""
    clf = pipeline.named_steps["clf"]
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X)
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        # Scale decision scores through a sigmoid for a demo-friendly confidence.
        scores = pipeline.decision_function(X)
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            return np.stack([1.0 - p1, p1], axis=1)
    pred = pipeline.predict(X)
    proba = np.zeros((len(pred), 2), dtype=np.float64)
    for i, y in enumerate(pred):
        proba[i, int(y)] = 1.0
    return proba


def linear_weights_for_cam(pipeline: Pipeline) -> Optional[np.ndarray]:
    """
    Return (2, 512) class weights in embedding space, or None if not linear.

    For a Pipeline(scaler, linear_clf):
        z = (x - mean) / scale
        logit = z @ coef.T + intercept
        equivalent raw-embedding weights = coef / scale
    """
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    coef = np.asarray(clf.coef_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    if coef.shape[0] == 1:
        # Binary: coef is for class 1 (not_good). Class 0 is the opposite.
        w1 = coef[0] / scale
        weights = np.stack([-w1, w1], axis=0)
    else:
        weights = coef / scale[None, :]
    return weights.astype(np.float32)


def save_checkpoint(payload: dict, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path = MODEL_PATH, map_location="cpu") -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path.as_posix()}. Run: python train.py"
        )
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_summary(ckpt: dict) -> dict:
    return {
        "approach": ckpt.get("approach"),
        "backbone": ckpt.get("backbone", BACKBONE_NAME),
        "image_size": ckpt.get("image_size", IMAGE_SIZE),
        "use_crop": ckpt.get("use_crop"),
        "selected_model": ckpt.get("selected_model"),
        "n_train_images": ckpt.get("n_train_images"),
        "metrics": ckpt.get("metrics"),
        "warnings": ckpt.get("warnings", []),
    }
