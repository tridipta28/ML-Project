"""
================================================================================
MODULE 4 & 5: Evaluation + Visualization
================================================================================
Covers:
  - IoU, mIoU, Dice Coefficient, Pixel Accuracy, Hausdorff Distance
  - Per-class metric breakdown
  - Metric comparison plots across models
  - Segmentation overlay visualizations (image / GT / prediction)
  - Critical analysis: success vs failure case detection
================================================================================
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")                       # headless / Colab safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff

from dataset import CITYSCAPES_CLASSES, ID_TO_COLOUR, NUM_CLASSES

IGNORE_INDEX = 255


# ═════════════════════════════════════════════════════════════════════════════
# Core Metric Implementation
# ═════════════════════════════════════════════════════════════════════════════

class SegmentationMetrics:
    """
    Accumulates confusion-matrix statistics over a dataset and
    computes standard semantic segmentation metrics:
      - Pixel Accuracy (PA)
      - Mean Pixel Accuracy (mPA)
      - IoU per class
      - Mean IoU (mIoU)
      - Dice Coefficient per class
      - Mean Dice
    Hausdorff Distance is computed separately (expensive).
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 ignore_index: int = IGNORE_INDEX):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Clear accumulated statistics."""
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes,
                                       dtype=torch.long)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds:   (B, H, W) predicted class IDs — already argmax'd
        targets: (B, H, W) ground-truth class IDs
        """
        # Flatten and mask out ignored pixels
        preds_flat   = preds.flatten()
        targets_flat = targets.flatten()
        valid        = targets_flat != self.ignore_index
        preds_flat   = preds_flat[valid]
        targets_flat = targets_flat[valid]

        # Accumulate into confusion matrix
        indices = self.num_classes * targets_flat + preds_flat
        conf    = torch.bincount(indices,
                                 minlength=self.num_classes ** 2)
        self.conf_matrix += conf.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from the accumulated confusion matrix."""
        C  = self.conf_matrix.float()
        tp = C.diagonal()                       # true positives per class
        fp = C.sum(dim=0) - tp                  # false positives
        fn = C.sum(dim=1) - tp                  # false negatives

        # ── IoU per class ─────────────────────────────────────────────────
        union = tp + fp + fn
        iou   = torch.where(union > 0, tp / (union + 1e-10),
                             torch.zeros_like(tp))

        # Only average over classes that appear in ground truth
        valid_classes = (C.sum(dim=1) > 0)
        miou = iou[valid_classes].mean().item()

        # ── Dice per class ────────────────────────────────────────────────
        dice = torch.where(
            (2 * tp + fp + fn) > 0,
            2 * tp / (2 * tp + fp + fn + 1e-10),
            torch.zeros_like(tp)
        )
        mean_dice = dice[valid_classes].mean().item()

        # ── Pixel Accuracy ────────────────────────────────────────────────
        total_correct = tp.sum().item()
        total_pixels  = C.sum().item()
        pa = total_correct / (total_pixels + 1e-10)

        # ── Mean Pixel Accuracy ───────────────────────────────────────────
        per_class_acc = torch.where(
            C.sum(dim=1) > 0,
            tp / (C.sum(dim=1) + 1e-10),
            torch.zeros_like(tp)
        )
        mpa = per_class_acc[valid_classes].mean().item()

        return {
            "mIoU":              miou,
            "mean_dice":         mean_dice,
            "pixel_accuracy":    pa,
            "mean_pixel_acc":    mpa,
            "iou_per_class":     iou.tolist(),
            "dice_per_class":    dice.tolist(),
            "per_class_acc":     per_class_acc.tolist(),
            "valid_classes":     valid_classes.tolist(),
        }

    def print_table(self):
        """Pretty-print per-class IoU alongside class names."""
        results = self.compute()
        ious    = results["iou_per_class"]
        print(f"\n{'─'*50}")
        print(f"{'Class':<18} {'IoU':>8} {'Dice':>8} {'Acc':>8}")
        print(f"{'─'*50}")
        for cls in CITYSCAPES_CLASSES:
            name, tid, _ = cls
            if results["valid_classes"][tid]:
                print(f"{name:<18} {ious[tid]:>8.4f} "
                      f"{results['dice_per_class'][tid]:>8.4f} "
                      f"{results['per_class_acc'][tid]:>8.4f}")
        print(f"{'─'*50}")
        print(f"{'mIoU':<18} {results['mIoU']:>8.4f}")
        print(f"{'Mean Dice':<18} {results['mean_dice']:>8.4f}")
        print(f"{'Pixel Acc':<18} {results['pixel_accuracy']:>8.4f}")
        print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Hausdorff Distance (expensive — run on a subset)
# ─────────────────────────────────────────────────────────────────────────────

def hausdorff_distance_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, float]:
    """
    Compute 95th-percentile Hausdorff Distance per class.
    Measures boundary quality — low value means crisp, well-aligned boundaries.

    pred, target: (H, W) numpy arrays of class IDs.
    """
    hd_per_class = {}
    for c in range(num_classes):
        p_mask = (pred   == c).astype(np.uint8)
        g_mask = (target == c).astype(np.uint8)
        if g_mask.sum() == 0:
            continue    # class absent in GT

        # Extract boundary coordinates
        p_coords = np.column_stack(np.where(p_mask > 0))
        g_coords = np.column_stack(np.where(g_mask > 0))

        if len(p_coords) == 0 or len(g_coords) == 0:
            hd_per_class[CITYSCAPES_CLASSES[c][0]] = float("inf")
            continue

        # Directed Hausdorff both ways → symmetric
        hd_pg = directed_hausdorff(p_coords, g_coords)[0]
        hd_gp = directed_hausdorff(g_coords, p_coords)[0]
        hd_per_class[CITYSCAPES_CLASSES[c][0]] = max(hd_pg, hd_gp)

    return hd_per_class


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    model_results: Dict[str, Dict],
    save_path: str = "outputs/model_comparison.png",
):
    """
    Bar chart comparing mIoU, Dice, and Pixel Accuracy across multiple models.
    model_results = {"ModelName": {"mIoU": 0.6, "mean_dice": 0.7, ...}, ...}
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    models  = list(model_results.keys())
    metrics = ["mIoU", "mean_dice", "pixel_accuracy"]
    labels  = ["mIoU", "Mean Dice", "Pixel Accuracy"]
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label, colour) in enumerate(zip(metrics, labels, colours)):
        vals = [model_results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=colour, edgecolor="white", linewidth=0.5)
        # Value annotations on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Semantic Segmentation — Model Comparison", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved model comparison → {save_path}")


def plot_per_class_iou(
    iou_per_class: List[float],
    model_name: str = "Model",
    save_path: str = "outputs/per_class_iou.png",
):
    """Horizontal bar chart of IoU for all 19 Cityscapes classes."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    class_names = [cls[0] for cls in CITYSCAPES_CLASSES]
    colours_rgb = [np.array(cls[2]) / 255.0 for cls in CITYSCAPES_CLASSES]

    fig, ax = plt.subplots(figsize=(8, 8))
    y_pos = np.arange(len(class_names))
    bars  = ax.barh(y_pos, iou_per_class, color=colours_rgb,
                    edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("IoU", fontsize=11)
    ax.set_title(f"Per-Class IoU — {model_name}", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.axvline(np.mean(iou_per_class), color="red", linestyle="--",
               linewidth=1, label=f"mIoU={np.mean(iou_per_class):.3f}")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, iou_per_class):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved per-class IoU → {save_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = "outputs/training_curves.png",
):
    """Plot train/val loss and mIoU over epochs."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss",
             color="#4C72B0", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], label="Val Loss",
             color="#DD8452", linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_miou"], label="Val mIoU",
             color="#55A868", linewidth=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU")
    ax2.set_title("Validation mIoU")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Training curves → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Visualization: Segmentation Overlays
# ═════════════════════════════════════════════════════════════════════════════

def label_to_rgb(label_map: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) label ID map to an (H, W, 3) RGB colour map.
    Pixels with ignore_index (255) are rendered as black.
    """
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cls in CITYSCAPES_CLASSES:
        name, tid, colour = cls
        rgb[label_map == tid] = colour
    return rgb


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for display.
    tensor: (C, H, W) float tensor, normalised.
    Returns: (H, W, C) uint8 numpy array.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def visualize_prediction(
    image:     torch.Tensor,        # (C, H, W) normalised
    gt_mask:   torch.Tensor,        # (H, W)
    pred_mask: torch.Tensor,        # (H, W)
    save_path: Optional[str] = None,
    title:     str = "Segmentation Result",
    alpha:     float = 0.5,
) -> None:
    """
    Display or save a 3-panel figure:
      [Original Image] | [Ground Truth] | [Prediction]
    with a shared class colour legend.
    """
    img_np   = denormalize(image)
    gt_rgb   = label_to_rgb(gt_mask.numpy())
    pred_rgb = label_to_rgb(pred_mask.numpy())

    # Overlay: blend segmentation colour map over original image
    gt_overlay   = (alpha * gt_rgb   + (1 - alpha) * img_np).astype(np.uint8)
    pred_overlay = (alpha * pred_rgb + (1 - alpha) * img_np).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img_np);         axes[0].set_title("Original Image")
    axes[1].imshow(gt_overlay);     axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_overlay);   axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")

    # Legend patches
    patches = [
        mpatches.Patch(color=np.array(cls[2]) / 255.0, label=cls[0])
        for cls in CITYSCAPES_CLASSES
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=10, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[Viz] Saved → {save_path}")
    else:
        plt.show()


def batch_visualize(
    model:     torch.nn.Module,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
    n_samples: int = 8,
    save_dir:  str = "outputs/visualizations",
):
    """Run model on first n_samples from loader and save overlay figures."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device)
            targets = batch["label"]
            paths   = batch["path"]

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(dim=1).cpu()

            for i in range(images.size(0)):
                stem = Path(paths[i]).stem
                visualize_prediction(
                    images[i].cpu(), targets[i], preds[i],
                    save_path=f"{save_dir}/{stem}.png",
                    title=Path(paths[i]).stem,
                )
                count += 1
                if count >= n_samples:
                    return


# ═════════════════════════════════════════════════════════════════════════════
# Critical Analysis: Success / Failure Case Detection
# ═════════════════════════════════════════════════════════════════════════════

# Classes typically large / easy to segment → success
LARGE_CLASSES   = {"road", "building", "vegetation", "sky", "sidewalk"}
# Classes typically small / rare / occluded → failure
SMALL_CLASSES   = {"rider", "motorcycle", "bicycle", "traffic light",
                   "traffic sign", "person"}


def analyse_cases(
    preds:   np.ndarray,      # (H, W)
    targets: np.ndarray,      # (H, W)
    image:   np.ndarray,      # (H, W, 3) for saving
    img_name: str = "sample",
    iou_threshold_success: float = 0.7,
    iou_threshold_failure: float = 0.3,
    save_dir: str = "outputs/analysis",
) -> Dict:
    """
    Classify each class in one image as success or failure,
    log metadata, and save annotated thumbnails.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    per_class_iou = {}
    for cls in CITYSCAPES_CLASSES:
        name, tid, _ = cls
        p = (preds == tid).astype(np.uint8)
        g = (targets == tid).astype(np.uint8)
        if g.sum() == 0:
            continue
        inter = (p & g).sum()
        union = (p | g).sum()
        per_class_iou[name] = float(inter) / (float(union) + 1e-10)

    successes = {k: v for k, v in per_class_iou.items()
                 if v >= iou_threshold_success and k in LARGE_CLASSES}
    failures  = {k: v for k, v in per_class_iou.items()
                 if v <  iou_threshold_failure and k in SMALL_CLASSES}

    report = {
        "image":       img_name,
        "per_class_iou": per_class_iou,
        "successes":   successes,
        "failures":    failures,
    }

    # Log summary
    if successes:
        print(f"  ✓ Success [{img_name}]: "
              + ", ".join(f"{k}={v:.2f}" for k, v in successes.items()))
    if failures:
        print(f"  ✗ Failure [{img_name}]: "
              + ", ".join(f"{k}={v:.2f}" for k, v in failures.items()))

    return report


def runtime_vs_accuracy_plot(
    model_names:    List[str],
    miou_scores:    List[float],
    fps_scores:     List[float],
    param_millions: List[float],
    save_path: str = "outputs/efficiency_plot.png",
):
    """
    Scatter plot: x=FPS (↑ better, faster inference)
                  y=mIoU (↑ better accuracy)
                  bubble size = parameter count
    Shows the classic accuracy–efficiency trade-off.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    colours = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for i, (name, miou, fps, params) in enumerate(
        zip(model_names, miou_scores, fps_scores, param_millions)
    ):
        sc = ax.scatter(fps, miou, s=params * 5, color=colours[i],
                        alpha=0.8, edgecolors="white", linewidths=0.5,
                        zorder=3, label=name)
        ax.annotate(name, (fps, miou),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=colours[i])

    ax.set_xlabel("Inference Speed (FPS)", fontsize=11)
    ax.set_ylabel("mIoU (%)", fontsize=11)
    ax.set_title("Accuracy vs. Efficiency Trade-off\n"
                 "(bubble size = parameter count)", fontsize=12)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Efficiency trade-off → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Full evaluation pass on a loader
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model:      torch.nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    model_name: str = "model",
    save_dir:   str = "outputs",
    compute_hd: bool = False,        # Hausdorff is expensive — opt-in
    n_vis:      int = 4,
) -> Dict:
    """
    Full evaluation:
      1. Accumulate confusion matrix over all validation batches.
      2. Compute all metrics.
      3. Save per-class IoU chart.
      4. Save visualizations for first n_vis images.
      5. Run critical analysis on the first 20 images.
      6. Measure inference FPS.
      Returns metrics dict.
    """
    print(f"\n{'─'*50}")
    print(f"  Evaluating: {model_name}")
    print(f"{'─'*50}")
    model.eval()
    metrics    = SegmentationMetrics(NUM_CLASSES)
    analysis   = []
    vis_count  = 0
    t_start    = time.time()
    n_images   = 0

    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["label"]
        paths   = batch["path"]

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = outputs.argmax(dim=1).cpu()

        metrics.update(preds, targets)
        n_images += images.size(0)

        # Visualize first n_vis images
        if vis_count < n_vis:
            for i in range(min(images.size(0), n_vis - vis_count)):
                stem = Path(paths[i]).stem
                visualize_prediction(
                    images[i].cpu(), targets[i], preds[i],
                    save_path=f"{save_dir}/vis/{model_name}_{stem}.png",
                    title=f"{model_name} — {stem}",
                )
                vis_count += 1

        # Critical analysis on first 20
        if len(analysis) < 20:
            for i in range(images.size(0)):
                img_np = denormalize(images[i].cpu())
                report = analyse_cases(
                    preds[i].numpy(), targets[i].numpy(),
                    img_np, img_name=Path(paths[i]).stem,
                    save_dir=f"{save_dir}/analysis",
                )
                analysis.append(report)

    elapsed = time.time() - t_start
    fps     = n_images / elapsed

    results = metrics.compute()
    results["fps"]        = fps
    results["model_name"] = model_name

    metrics.print_table()
    print(f"  FPS: {fps:.1f}")

    # Save per-class IoU chart
    plot_per_class_iou(
        results["iou_per_class"],
        model_name=model_name,
        save_path=f"{save_dir}/iou_{model_name}.png",
    )

    # Save analysis JSON
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/analysis_{model_name}.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Hausdorff Distance (expensive — just first batch)
    if compute_hd:
        for batch in loader:
            images  = batch["image"].to(device)
            targets = batch["label"]
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(dim=1).cpu()
            hd = hausdorff_distance_per_class(preds[0].numpy(),
                                               targets[0].numpy())
            results["hausdorff"] = hd
            print("  Hausdorff (first image):", hd)
            break

    return results
