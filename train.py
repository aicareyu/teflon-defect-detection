"""
Train a Good vs Not-Good classifier on a very small industrial dataset.

Usage (from the project root):
    python train.py

Strategy
--------
1. Load image-level labels from dataset/good (0) and dataset/not_good (1).
2. Optional OpenCV product crop to reduce background overfitting.
3. Extract frozen ResNet-18 ImageNet embeddings.
4. Compare LogisticRegression / Linear SVM / RBF SVM with leave-one-out CV.
5. Optionally compare a frozen-backbone + linear-head PyTorch model.
6. Retrain the winner on ALL images and save models/defect_classifier.pth.

Leave-one-out scores on ~15 images are NOT statistically reliable. They are
reported only as a demo diagnostic.
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CLASS_MAP_PATH,
    CLASS_TO_IDX,
    FINE_TUNE_BATCH_SIZE,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    FINE_TUNE_WEIGHT_DECAY,
    IDX_TO_CLASS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    POC_DISCLAIMER,
    POSE_WARNING,
    RANDOM_SEED,
    SMALL_DATA_WARNING,
    USE_PRODUCT_CROP,
)
from src.dataset import collect_samples  # noqa: E402
from src.model import (  # noqa: E402
    ResNet18Encoder,
    TransferHeadClassifier,
    extract_embeddings,
    get_device,
    make_sklearn_candidates,
    save_checkpoint,
)
from src.preprocess import build_train_augment, imagenet_tensor, prepare_from_path  # noqa: E402


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0, pos_label=1)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_samples": int(len(y_true)),
        "n_positive_not_good": int((y_true == 1).sum()),
        "n_negative_good": int((y_true == 0).sum()),
    }


def load_processed_images(paths, use_crop: bool):
    originals = []
    displays = []
    nets = []
    crop_stats = []
    for p in paths:
        original, display, net, info = prepare_from_path(p, use_crop=use_crop)
        originals.append(original)
        displays.append(display)
        nets.append(net)
        crop_stats.append(info["applied"])
    tensors = torch.cat([imagenet_tensor(n) for n in nets], dim=0)
    return originals, displays, nets, tensors, crop_stats


def expand_with_mild_augs(encoder, nets, y, device, n_extra: int = 3):
    """
    One original embedding plus a few mild augmentations per image.

    Augmentations stay conservative (flip, small affine, brightness/contrast,
    slight blur) so the physical defect is not invented or destroyed.
    Groups keep all views of the same photo together for honest leave-one-image-out.
    """
    aug = build_train_augment()
    chunks, labels, groups = [], [], []
    for i, net in enumerate(nets):
        views = [imagenet_tensor(net)]
        pil = Image.fromarray(net)
        for _ in range(n_extra):
            views.append(aug(pil).unsqueeze(0))
        batch = torch.cat(views, dim=0)
        emb, _ = extract_embeddings(encoder, batch, device)
        chunks.append(emb)
        labels.extend([int(y[i])] * len(views))
        groups.extend([i] * len(views))
    return (
        np.concatenate(chunks, axis=0),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


def grouped_loocv_sklearn(pipeline, X, y, groups) -> dict:
    """Leave one *source image* out; train on other images including their augs."""
    y_true, y_pred = [], []
    for gid in np.unique(groups):
        test_mask = groups == gid
        train_mask = ~test_mask
        test_idx = int(np.where(test_mask)[0][0])  # original view is stored first
        est = clone(pipeline)
        est.fit(X[train_mask], y[train_mask])
        y_pred.append(int(est.predict(X[test_idx : test_idx + 1])[0]))
        y_true.append(int(y[test_idx]))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = binary_metrics(y_true, y_pred)
    metrics["protocol"] = "leave_one_image_out_with_train_augs"
    return metrics


def save_confusion_matrix_png(cm, path: Path, n_eval: int) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred GOOD", "pred NOT GOOD"])
    ax.set_yticks([0, 1], labels=["true GOOD", "true NOT GOOD"])
    for (r, c), val in np.ndenumerate(cm):
        ax.text(c, r, str(val), ha="center", va="center", color="black", fontsize=14)
    ax.set_title(f"Confusion matrix (n={n_eval} held-out images)\nNot production accuracy")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def maybe_stratified_holdout(y: np.ndarray, min_per_class: int = 2):
    counts = np.bincount(y, minlength=2)
    if counts.min() < min_per_class + 1:
        return None, (
            "Dataset is too small for a reliable stratified validation split "
            f"(class counts = {counts.tolist()}). Leave-one-out is used instead. "
            "Metrics from this many images are not statistically reliable."
        )
    try:
        idx = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            idx, test_size=0.25, stratify=y, random_state=RANDOM_SEED
        )
        if len(val_idx) < 3:
            return None, (
                "Validation split would contain fewer than 3 images. "
                "Leave-one-out is used instead."
            )
        return (train_idx, val_idx), None
    except ValueError as exc:
        return None, f"Could not create a stratified split ({exc}). Using leave-one-out."


def train_finetune_head(
    tensors: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple:
    """Frozen ResNet-18 + linear head. Augmentation applied on-the-fly."""
    model = TransferHeadClassifier().to(device)
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINE_TUNE_LR,
        weight_decay=FINE_TUNE_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    # Mild input jitter only: the backbone stays frozen. Aggressive geometric
    # warps are avoided so a real defect is not cropped out of the frame.
    x_train = tensors[train_idx]
    y_train = torch.from_numpy(y[train_idx]).long()
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(
        dataset,
        batch_size=min(FINE_TUNE_BATCH_SIZE, len(dataset)),
        shuffle=True,
        drop_last=False,
    )

    model.train()
    # Unfreeze only fc (already done). Train a modest number of epochs.
    for _epoch in range(FINE_TUNE_EPOCHS):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # Light noise on embeddings-scale activations: small input jitter
            # after normalisation is equivalent to a tiny brightness shift.
            if random.random() < 0.5:
                xb = xb + 0.02 * torch.randn_like(xb)
            xb = torch.flip(xb, dims=[3]) if random.random() < 0.5 else xb
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(tensors[val_idx].to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
    metrics = binary_metrics(y[val_idx], pred)
    metrics["protocol"] = "stratified_holdout_finetune"
    metrics["n_val"] = int(len(val_idx))
    metrics["n_train"] = int(len(train_idx))
    return model, metrics


def main() -> int:
    set_seed()
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 72)
    print("  PLASTIC FITTING DEFECT INSPECTION - TRAINING")
    print("=" * 72)
    print(SMALL_DATA_WARNING)
    print(POC_DISCLAIMER)
    print(POSE_WARNING)
    print()

    paths, y, _names = collect_samples()
    n = len(paths)
    n_good = int((y == 0).sum())
    n_ng = int((y == 1).sum())
    print(f"Loaded {n} images  |  good={n_good}  not_good={n_ng}")
    print(f"Product crop enabled: {USE_PRODUCT_CROP}")

    if n < 6:
        print("ERROR: Need at least a handful of images in each class.")
        return 1

    split_warn = None
    if n < 30:
        split_warn = (
            f"Only {n} labelled images. Cross-validation scores will be noisy "
            "and must not be treated as production accuracy."
        )
        print("WARNING:", split_warn)

    device = get_device()
    print(f"Device: {device}")
    print("Loading pretrained ResNet-18 (ImageNet)...")
    encoder = ResNet18Encoder()

    _orig, _disp, _nets, tensors, crop_stats = load_processed_images(
        paths, use_crop=USE_PRODUCT_CROP
    )
    print(
        f"Foreground crop applied on {sum(crop_stats)}/{n} images "
        "(failed crops keep the full frame)."
    )

    print("Extracting frozen embeddings (original + mild training views)...")
    X, y_exp, groups = expand_with_mild_augs(encoder, _nets, y, device, n_extra=3)
    print(f"Embedding table: {X.shape[0]} views from {n} images, dim={X.shape[1]}")

    results = []
    for name, pipe in make_sklearn_candidates(RANDOM_SEED):
        print(f"\nLeave-one-image-out evaluation: {name}")
        metrics = grouped_loocv_sklearn(pipe, X, y_exp, groups)
        print(
            f"  acc={metrics['accuracy']:.3f}  "
            f"prec={metrics['precision']:.3f}  "
            f"rec={metrics['recall']:.3f}  "
            f"f1={metrics['f1']:.3f}  "
            f"n={metrics['n_samples']} (held-out images)"
        )
        print(f"  confusion_matrix [rows=true good/ng, cols=pred]: {metrics['confusion_matrix']}")
        results.append((metrics["f1"], name, pipe, metrics, "embedding_sklearn"))

    holdout, holdout_msg = maybe_stratified_holdout(y)
    ft_metrics = None
    ft_model = None
    if holdout is None:
        print("\nTransfer-learning head: skipped for comparison.")
        print(" ", holdout_msg)
    else:
        train_idx, val_idx = holdout
        print(
            f"\nTransfer-learning head on stratified holdout "
            f"(train={len(train_idx)}, val={len(val_idx)})..."
        )
        print(
            "WARNING: A validation set of this size cannot support a strong "
            "accuracy claim."
        )
        ft_model, ft_metrics = train_finetune_head(
            tensors, y, device, train_idx, val_idx
        )
        print(
            f"  acc={ft_metrics['accuracy']:.3f}  "
            f"prec={ft_metrics['precision']:.3f}  "
            f"rec={ft_metrics['recall']:.3f}  "
            f"f1={ft_metrics['f1']:.3f}  "
            f"n_val={ft_metrics['n_val']}"
        )
        results.append(
            (ft_metrics["f1"], "finetune_fc", ft_model, ft_metrics, "finetune_resnet18")
        )
        if holdout_msg:
            print(" ", holdout_msg)

    results.sort(key=lambda t: t[0], reverse=True)

    sklearn_rows = [r for r in results if r[4] == "embedding_sklearn"]
    ft_rows = [r for r in results if r[4] == "finetune_resnet18"]
    best_sk = max(sklearn_rows, key=lambda t: t[0]) if sklearn_rows else None
    best_ft = max(ft_rows, key=lambda t: t[0]) if ft_rows else None

    # On ~15 images a 3–4 image holdout can randomly beat leave-one-out.
    # Prefer frozen embeddings + linear classifier unless the head is evaluated
    # on a comparable number of images AND is clearly better.
    prefer_msg = None
    if best_sk is not None and best_ft is not None:
        ft_n = int(best_ft[3].get("n_val") or best_ft[3].get("n_samples") or 0)
        sk_n = int(best_sk[3].get("n_samples") or 0)
        if ft_n >= max(8, sk_n) and best_ft[0] > best_sk[0] + 0.05:
            chosen = best_ft
        else:
            chosen = best_sk
            prefer_msg = (
                "Preferred pretrained embeddings + linear classifier. "
                f"The fine-tune head reported F1={best_ft[0]:.3f} on only "
                f"{ft_n} validation image(s), which is not a fair comparison."
            )
    else:
        chosen = results[0]

    best_f1, best_name, best_obj, best_metrics, approach = chosen
    print("\n" + "-" * 72)
    print(f"Selected approach: {approach} / {best_name}  (eval F1={best_f1:.3f})")
    if prefer_msg:
        print(prefer_msg)
    print(
        "This ranking is based on a tiny sample and is not statistically reliable."
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "approach": approach,
        "selected_model": best_name,
        "backbone": "resnet18",
        "image_size": 224,
        "use_crop": USE_PRODUCT_CROP,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
        "n_train_images": n,
        "n_good": n_good,
        "n_not_good": n_ng,
        "metrics": best_metrics,
        "all_model_scores": [
            {"name": name, "approach": appr, "metrics": met}
            for _f1, name, _obj, met, appr in results
        ],
        "warnings": [
            SMALL_DATA_WARNING,
            POC_DISCLAIMER,
            POSE_WARNING,
        ]
        + ([split_warn] if split_warn else [])
        + ([holdout_msg] if holdout_msg else [])
        + ([prefer_msg] if prefer_msg else []),
        "image_paths": [p.as_posix() for p in paths],
        "labels": y.tolist(),
    }

    # Retrain winner on ALL labelled images for the demo checkpoint.
    if approach == "embedding_sklearn":
        final_pipe = clone(best_obj)
        final_pipe.fit(X, y_exp)
        payload["sklearn_pipeline"] = final_pipe
        payload["encoder_name"] = "resnet18"
    else:
        # Retrain the head on all tensors for a few extra epochs.
        all_idx = np.arange(n)
        ft_model, _ = train_finetune_head(tensors, y, device, all_idx, all_idx[: max(1, n // 5)])
        payload["finetune_state_dict"] = ft_model.state_dict()

    cm_path = MODELS_DIR / "confusion_matrix.png"
    save_confusion_matrix_png(
        np.asarray(best_metrics["confusion_matrix"]),
        cm_path,
        best_metrics.get("n_samples", best_metrics.get("n_val", n)),
    )
    payload["confusion_matrix_plot"] = cm_path.as_posix()

    save_checkpoint(payload, MODEL_PATH)

    class_map = {
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
        "display": {"0": "GOOD", "1": "NOT GOOD"},
        "model_path": MODEL_PATH.as_posix(),
        "approach": approach,
        "selected_model": best_name,
    }
    CLASS_MAP_PATH.write_text(json.dumps(class_map, indent=2), encoding="utf-8")

    serializable = {
        k: v
        for k, v in payload.items()
        if k not in {"sklearn_pipeline", "finetune_state_dict"}
    }
    METRICS_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(f"\nSaved model:        {MODEL_PATH.as_posix()}")
    print(f"Saved class map:    {CLASS_MAP_PATH.as_posix()}")
    print(f"Saved confusion matrix: {cm_path.as_posix()}")
    print(f"Saved metrics JSON: {METRICS_PATH.as_posix()}")
    from src.gallery import build_gallery_from_dataset

    gal = build_gallery_from_dataset()
    print(f"Saved embedding gallery: {gal}")
    print("\nConfusion matrix interpretation (LOOCV or tiny holdout):")
    print("  rows = true class [GOOD, NOT GOOD]")
    print("  cols = predicted  [GOOD, NOT GOOD]")
    print("Do not report this as production accuracy.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
