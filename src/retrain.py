"""Fast embedding-classifier refit after new live examples are added."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.base import clone

from src.config import (
    CLASS_MAP_PATH,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    POC_DISCLAIMER,
    POSE_WARNING,
    SMALL_DATA_WARNING,
    USE_PRODUCT_CROP,
)
from src.dataset import collect_samples
from src.model import (
    ResNet18Encoder,
    extract_embeddings,
    get_device,
    make_sklearn_candidates,
    save_checkpoint,
)
from src.preprocess import imagenet_tensor, prepare_from_path


def retrain_embedding_classifier() -> dict:
    """
    Frozen ResNet-18 embeddings + logistic regression on the current folders.

    Skips leave-one-out and PyTorch fine-tune so a live 'teach' click stays short.
    """
    paths, y, _names = collect_samples()
    n = len(paths)
    n_good = int((y == 0).sum())
    n_ng = int((y == 1).sum())
    device = get_device()
    encoder = ResNet18Encoder()
    nets = []
    for p in paths:
        _o, _d, net, _info = prepare_from_path(p, use_crop=USE_PRODUCT_CROP)
        nets.append(net)
    tensors = torch.cat([imagenet_tensor(n) for n in nets], dim=0)
    X, _maps = extract_embeddings(encoder, tensors, device)
    pipe = clone(make_sklearn_candidates()[0][1])
    pipe.fit(X, y)
    payload = {
        "approach": "embedding_sklearn",
        "selected_model": "logreg",
        "backbone": "resnet18",
        "image_size": 224,
        "use_crop": USE_PRODUCT_CROP,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
        "n_train_images": n,
        "n_good": n_good,
        "n_not_good": n_ng,
        "metrics": {
            "protocol": "refit_all_images_no_holdout",
            "n_samples": n,
            "note": "Refit after live teach. No held-out accuracy is reported.",
        },
        "warnings": [SMALL_DATA_WARNING, POC_DISCLAIMER, POSE_WARNING],
        "sklearn_pipeline": pipe,
        "encoder_name": "resnet18",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(payload, MODEL_PATH)
    CLASS_MAP_PATH.write_text(
        json.dumps(
            {
                "class_to_idx": CLASS_TO_IDX,
                "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
                "n_good": n_good,
                "n_not_good": n_ng,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    serializable = {k: v for k, v in payload.items() if k != "sklearn_pipeline"}
    METRICS_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return {"n": n, "n_good": n_good, "n_not_good": n_ng, "path": MODEL_PATH.as_posix()}
