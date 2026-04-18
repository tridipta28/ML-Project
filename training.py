"""
================================================================================
MODULE 3: Training Pipeline
================================================================================
Covers:
  - Loss functions: Dice Loss, Focal Loss, Weighted Cross-Entropy, Combined
  - Optimizers: Adam, SGD with momentum + LR schedulers
  - Mixed precision training (torch.cuda.amp)
  - Training loop with validation
  - TensorBoard logging
  - Checkpoint saving / loading
================================================================================
"""

import os
import time
import math
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataset import NUM_CLASSES


# ═════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ═════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Maximises the overlap between prediction and ground truth.
    Naturally handles class imbalance — rare foreground classes contribute
    proportionally more than abundant background.

    dice = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C, H, W) — raw model outputs (pre-softmax)
        targets: (B, H, W)    — ground-truth class IDs
        """
        B, C, H, W = logits.shape

        # Create valid pixel mask (exclude ignore_index=255)
        valid = (targets != self.ignore_index)

        # One-hot encode targets: (B, C, H, W)
        tgt_one_hot = torch.zeros_like(logits)
        tgt_clamped = targets.clone()
        tgt_clamped[~valid] = 0                 # map ignore pixels to class 0 temporarily
        tgt_one_hot.scatter_(1, tgt_clamped.unsqueeze(1), 1.0)
        tgt_one_hot = tgt_one_hot * valid.unsqueeze(1).float()

        probs = logits.softmax(dim=1)

        dice_per_class = []
        for c in range(C):
            p = probs[:, c]            # (B, H, W)
            g = tgt_one_hot[:, c]      # (B, H, W)
            inter = (p * g).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
            dice_per_class.append(
                (1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)).mean()
            )

        return torch.stack(dice_per_class).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017).
    Down-weights easy negatives so training focuses on hard examples.
    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    α: class balancing weight (uniform or per-class)
    γ: focusing parameter — γ=0 → standard CE; γ=2 is typical
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha       # (C,) tensor or None
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape

        # Flatten spatial dims
        logits_flat  = logits.permute(0, 2, 3, 1).reshape(-1, C)   # (N, C)
        targets_flat = targets.reshape(-1)                           # (N,)

        # Mask out ignored pixels
        valid_mask   = targets_flat != self.ignore_index
        logits_flat  = logits_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

        if logits_flat.numel() == 0:
            return logits.sum() * 0.0  # zero-grad safe

        # Standard cross-entropy per pixel
        log_p = F.log_softmax(logits_flat, dim=1)
        ce    = F.nll_loss(log_p, targets_flat, reduction="none")   # (N,)

        # Probability of the correct class
        p_t  = torch.exp(-ce)

        # Focal weighting
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets_flat]
            loss = alpha_t * loss

        return loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy with per-class weights.
    Weights are computed from inverse class frequency on the training set,
    making rare classes (riders, motorcycles) contribute more to the loss.
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class CombinedLoss(nn.Module):
    """
    Weighted combination: w_ce * CE + w_dice * Dice + w_focal * Focal.
    Default weights are empirically tuned for Cityscapes.
    """
    def __init__(self,
                 w_ce: float    = 1.0,
                 w_dice: float  = 0.5,
                 w_focal: float = 0.5,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 255):
        super().__init__()
        self.w_ce    = w_ce
        self.w_dice  = w_dice
        self.w_focal = w_focal

        self.ce    = WeightedCrossEntropyLoss(class_weights, ignore_index)
        self.dice  = DiceLoss(ignore_index=ignore_index)
        self.focal = FocalLoss(gamma=2.0, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_ce    = self.ce(logits, targets)
        l_dice  = self.dice(logits, targets)
        l_focal = self.focal(logits, targets)
        total   = (self.w_ce    * l_ce +
                   self.w_dice  * l_dice +
                   self.w_focal * l_focal)
        return {
            "total": total, "ce": l_ce,
            "dice":  l_dice, "focal": l_focal,
        }


def compute_class_weights(dataset, num_classes: int = NUM_CLASSES,
                          ignore_index: int = 255) -> torch.Tensor:
    """
    Compute median-frequency class weights from a dataset subset.
    w_c = median(freq) / freq_c
    This up-weights rare classes without extreme imbalance.
    """
    print("[Weights] Computing class frequencies (sample 500 images)...")
    counts = torch.zeros(num_classes)
    n_samples = min(500, len(dataset))
    indices   = torch.randperm(len(dataset))[:n_samples]

    for idx in indices:
        label = dataset[int(idx)]["label"]
        for c in range(num_classes):
            counts[c] += (label == c).sum().item()

    freq     = counts / counts.sum()
    # Avoid division by zero for classes with zero frequency
    freq     = torch.clamp(freq, min=1e-6)
    median_f = torch.median(freq)
    weights  = median_f / freq
    weights  = torch.clamp(weights, max=10.0)   # cap to avoid instability
    print(f"[Weights] Min={weights.min():.3f}, Max={weights.max():.3f}, "
          f"Mean={weights.mean():.3f}")
    return weights


# ═════════════════════════════════════════════════════════════════════════════
# Optimizer & Scheduler Factory
# ═════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Build optimizer from config dict.
    cfg keys: type, lr, weight_decay, momentum (SGD only)
    Backbone and head can have different learning rates (common practice).
    """
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name or "layer" in name or "encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": cfg["lr"] * cfg.get("backbone_lr_mult", 0.1)},
        {"params": head_params,     "lr": cfg["lr"]},
    ]

    opt_type = cfg.get("type", "adam").lower()
    if opt_type == "adam":
        return torch.optim.Adam(param_groups,
                                lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 1e-4))
    elif opt_type == "adamw":
        return torch.optim.AdamW(param_groups,
                                 lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 1e-2))
    elif opt_type == "sgd":
        return torch.optim.SGD(param_groups,
                               lr=cfg["lr"],
                               momentum=cfg.get("momentum", 0.9),
                               weight_decay=cfg.get("weight_decay", 5e-4),
                               nesterov=True)
    raise ValueError(f"Unknown optimizer type: {opt_type}")


def build_scheduler(optimizer, cfg: dict, num_epochs: int):
    """
    Build learning rate scheduler.
    cfg.scheduler: "cosine" | "poly" | "step" | "warmup_cosine"
    """
    sched = cfg.get("scheduler", "cosine")

    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=cfg["lr"] * 1e-3
        )
    elif sched == "poly":
        # Polynomial decay: (1 - iter/max_iter)^power
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda e: (1 - e / num_epochs) ** 0.9
        )
    elif sched == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.get("milestones", [30, 60, 90]),
            gamma=0.1
        )
    elif sched == "warmup_cosine":
        warmup_steps = cfg.get("warmup_epochs", 5)
        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return epoch / max(warmup_steps, 1)
            return 0.5 * (1 + math.cos(
                math.pi * (epoch - warmup_steps) / (num_epochs - warmup_steps)
            ))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {sched}")


# ═════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Encapsulates the full training lifecycle:
      - Forward pass with mixed precision
      - Backward pass + gradient clipping
      - Validation
      - TensorBoard logging
      - Best-model checkpoint saving
    """
    def __init__(
        self,
        model:       nn.Module,
        criterion:   nn.Module,
        optimizer:   torch.optim.Optimizer,
        scheduler,
        device:      torch.device,
        log_dir:     str = "runs/experiment",
        ckpt_dir:    str = "checkpoints",
        use_amp:     bool = True,
        grad_clip:   float = 1.0,
        aux_loss_wt: float = 0.4,    # weight for PSPNet auxiliary loss
    ):
        self.model       = model
        self.criterion   = criterion
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.use_amp     = use_amp and torch.cuda.is_available()
        self.grad_clip   = grad_clip
        self.aux_loss_wt = aux_loss_wt

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"[Trainer] TensorBoard → tensorboard --logdir {log_dir}")

        # Checkpoint dir
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(ckpt_dir)

        # Tracking
        self.best_miou  = 0.0
        self.best_epoch = 0
        self.history    = {"train_loss": [], "val_loss": [], "val_miou": []}

    # ─────────────────────────────────────────────────────────────────────
    def _train_epoch(self, loader) -> Dict[str, float]:
        """Run one training epoch. Returns avg loss dict."""
        self.model.train()
        total_losses = {"total": 0.0, "ce": 0.0, "dice": 0.0, "focal": 0.0}
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            images  = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward (with mixed precision) ────────────────────────────
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)

                # Handle PSPNet auxiliary output
                aux_loss = 0.0
                if isinstance(outputs, tuple):
                    outputs, aux_out = outputs
                    aux_losses = self.criterion(aux_out, targets)
                    aux_loss   = aux_losses["total"] * self.aux_loss_wt

                losses = self.criterion(outputs, targets)
                total_loss = losses["total"] + aux_loss

            # ── Backward ──────────────────────────────────────────────────
            self.scaler.scale(total_loss).backward()

            # Gradient clipping (prevents exploding gradients in transformers)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in total_losses:
                total_losses[k] += losses.get(k, torch.tensor(0.0)).item()
            n_batches += 1

            if batch_idx % 50 == 0:
                print(f"  Step {batch_idx}/{len(loader)} | "
                      f"loss={losses['total'].item():.4f}")

        return {k: v / n_batches for k, v in total_losses.items()}

    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _val_epoch(self, loader) -> Tuple[float, float]:
        """Validation epoch. Returns (avg_loss, mIoU)."""
        from evaluation import SegmentationMetrics
        self.model.eval()
        metrics   = SegmentationMetrics(NUM_CLASSES)
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            images  = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                losses = self.criterion(outputs, targets)

            total_loss += losses["total"].item()
            preds = outputs.argmax(dim=1)           # (B, H, W)
            metrics.update(preds.cpu(), targets.cpu())
            n_batches += 1

        avg_loss = total_loss / n_batches
        results  = metrics.compute()
        return avg_loss, results["mIoU"]

    # ─────────────────────────────────────────────────────────────────────
    def train(self, train_loader, val_loader, num_epochs: int = 100):
        """Full training loop with validation, logging, and checkpointing."""
        print(f"\n{'═'*60}")
        print(f"  Training for {num_epochs} epochs on {self.device}")
        print(f"  AMP={'ON' if self.use_amp else 'OFF'}")
        print(f"{'═'*60}\n")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            # ── Train ─────────────────────────────────────────────────────
            train_losses = self._train_epoch(train_loader)

            # ── Validate ──────────────────────────────────────────────────
            val_loss, val_miou = self._val_epoch(val_loader)

            # ── Scheduler step ────────────────────────────────────────────
            self.scheduler.step()
            lr = self.optimizer.param_groups[-1]["lr"]

            elapsed = time.time() - t0
            print(f"[Epoch {epoch:03d}/{num_epochs}] "
                  f"train_loss={train_losses['total']:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"mIoU={val_miou:.4f} | "
                  f"lr={lr:.2e} | {elapsed:.0f}s")

            # ── TensorBoard ───────────────────────────────────────────────
            self.writer.add_scalars("Loss", {
                "train": train_losses["total"], "val": val_loss
            }, epoch)
            self.writer.add_scalar("Loss/dice",  train_losses["dice"],  epoch)
            self.writer.add_scalar("Loss/focal", train_losses["focal"], epoch)
            self.writer.add_scalar("Loss/ce",    train_losses["ce"],    epoch)
            self.writer.add_scalar("Metrics/mIoU",     val_miou, epoch)
            self.writer.add_scalar("LR", lr, epoch)

            # ── History ───────────────────────────────────────────────────
            self.history["train_loss"].append(train_losses["total"])
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(val_miou)

            # ── Checkpoint (best mIoU) ─────────────────────────────────────
            if val_miou > self.best_miou:
                self.best_miou  = val_miou
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_miou, is_best=True)

            # Save periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_miou, is_best=False)

        print(f"\n✓ Training complete. Best mIoU={self.best_miou:.4f} "
              f"at epoch {self.best_epoch}")
        self.writer.close()
        return self.history

    # ─────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, epoch: int, miou: float, is_best: bool = False):
        name = "best.pth" if is_best else f"epoch_{epoch:03d}.pth"
        path = self.ckpt_dir / name
        torch.save({
            "epoch":      epoch,
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "miou":       miou,
            "history":    self.history,
        }, path)
        print(f"  [Checkpoint] Saved → {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_miou  = ckpt["miou"]
        self.history    = ckpt.get("history", self.history)
        print(f"[Checkpoint] Loaded epoch {ckpt['epoch']} "
              f"(mIoU={ckpt['miou']:.4f}) from {path}")
        return ckpt["epoch"]


# ═════════════════════════════════════════════════════════════════════════════
# Default Training Config
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Optimizer
    "optimizer": {
        "type":            "adamw",
        "lr":              6e-5,
        "weight_decay":    1e-2,
        "backbone_lr_mult": 0.1,
        "scheduler":       "warmup_cosine",
        "warmup_epochs":   5,
    },
    # Loss
    "loss": {
        "w_ce":    1.0,
        "w_dice":  0.5,
        "w_focal": 0.5,
    },
    # Training
    "num_epochs":    100,
    "batch_size":    4,
    "img_h":         512,
    "img_w":         1024,
    "use_amp":       True,
    "grad_clip":     1.0,
    "aux_loss_wt":   0.4,
}


if __name__ == "__main__":
    # Standalone loss test
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits  = torch.randn(2, NUM_CLASSES, 64, 128, device=device)
    targets = torch.randint(0, NUM_CLASSES, (2, 64, 128), device=device)
    targets[0, 0, 0] = 255    # inject an ignore pixel

    criterion = CombinedLoss()
    losses    = criterion(logits, targets)
    print("Loss breakdown:", {k: f"{v.item():.4f}" for k, v in losses.items()})
