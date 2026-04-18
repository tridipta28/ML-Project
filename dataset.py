"""
================================================================================
MODULE 1: Dataset Setup — Cityscapes Dataset Loader
================================================================================
Handles:
  - Dataset download instructions
  - Custom PyTorch Dataset class for Cityscapes
  - Preprocessing: resize, normalize, augmentations (flips, rotations, weather)
  - Train / Validation / Test splits
================================================================================
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────────────────────
# Cityscapes Class Definitions (19 training classes)
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (class_name, train_id, colour_RGB)
CITYSCAPES_CLASSES = [
    ("road",          0,  (128,  64, 128)),
    ("sidewalk",      1,  (244,  35, 232)),
    ("building",      2,  ( 70,  70,  70)),
    ("wall",          3,  (102, 102, 156)),
    ("fence",         4,  (190, 153, 153)),
    ("pole",          5,  (153, 153, 153)),
    ("traffic light", 6,  (250, 170,  30)),
    ("traffic sign",  7,  (220, 220,   0)),
    ("vegetation",    8,  (107, 142,  35)),
    ("terrain",       9,  (152, 251, 152)),
    ("sky",           10, ( 70, 130, 180)),
    ("person",        11, (220,  20,  60)),
    ("rider",         12, (255,   0,   0)),
    ("car",           13, (  0,   0, 142)),
    ("truck",         14, (  0,   0,  70)),
    ("bus",           15, (  0,  60, 100)),
    ("train",         16, (  0,  80, 100)),
    ("motorcycle",    17, (  0,   0, 230)),
    ("bicycle",       18, (119,  11,  32)),
]

# Flat lookup: train_id → RGB colour (used for visualisation)
ID_TO_COLOUR = {cls[1]: cls[2] for cls in CITYSCAPES_CLASSES}
NUM_CLASSES = 19

# Cityscapes raw label IDs → training IDs mapping
# Labels not in this map are ignored (train_id = 255)
LABEL_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}


# ─────────────────────────────────────────────────────────────────────────────
# Download Instructions
# ─────────────────────────────────────────────────────────────────────────────

def print_download_instructions():
    """
    Cityscapes requires registration at https://www.cityscapes-dataset.com/
    Downloads cannot be automated without credentials.
    This function prints a step-by-step guide.
    """
    instructions = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         CITYSCAPES DATASET — DOWNLOAD INSTRUCTIONS           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  1. Register at: https://www.cityscapes-dataset.com/         ║
    ║  2. Download the following packages:                         ║
    ║     • gtFine_trainvaltest.zip      (~241 MB) — annotations   ║
    ║     • leftImg8bit_trainvaltest.zip (~11 GB)  — images        ║
    ║  3. Extract to a local directory, e.g.:                      ║
    ║     /data/cityscapes/                                        ║
    ║       ├── gtFine/                                            ║
    ║       │   ├── train/  val/  test/                            ║
    ║       └── leftImg8bit/                                       ║
    ║           ├── train/  val/  test/                            ║
    ║  4. Set CITYSCAPES_ROOT in your config.                      ║
    ║                                                              ║
    ║  Alternative (auto download with credentials):               ║
    ║     pip install cityscapesscripts                            ║
    ║     csDownload -d /data/cityscapes                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(instructions)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(height: int = 512, width: int = 1024) -> A.Compose:
    """
    Training augmentation pipeline using albumentations.
    Applies geometric and photometric transforms.
    Weather effects (rain, fog) simulate real-world driving conditions.
    """
    return A.Compose([
        # --- Geometric ---
        A.RandomResizedCrop(height=height, width=width,
                            scale=(0.5, 1.0), ratio=(1.5, 2.5)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=10, p=0.5,
                           border_mode=0),        # reflect padding

        # --- Photometric ---
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),

        # --- Weather simulation ---
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
        A.RandomRain(slant_lower=-10, slant_upper=10,
                     drop_length=15, drop_width=1,
                     drop_color=(200, 200, 200), p=0.1),
        A.RandomSunFlare(num_flare_circles_lower=1,
                         num_flare_circles_upper=3, p=0.05),

        # --- Normalise & tensorise ---
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(height: int = 512, width: int = 1024) -> A.Compose:
    """Validation / test transform — only resize and normalise (no augmentation)."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Label Conversion Utility
# ─────────────────────────────────────────────────────────────────────────────

def convert_label_to_trainid(label: np.ndarray) -> np.ndarray:
    """
    Convert raw Cityscapes label IDs (0-33) to training IDs (0-18, 255).
    Pixels with no valid training class are set to 255 (ignored in loss).
    """
    train_mask = np.full(label.shape, 255, dtype=np.uint8)
    for src_id, train_id in LABEL_TO_TRAINID.items():
        train_mask[label == src_id] = train_id
    return train_mask


# ─────────────────────────────────────────────────────────────────────────────
# Cityscapes Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

class CityscapesDataset(Dataset):
    """
    PyTorch Dataset for the Cityscapes Semantic Segmentation benchmark.

    Directory structure expected:
        root/
          leftImg8bit/{split}/{city}/{city}_*_leftImg8bit.png
          gtFine/{split}/{city}/{city}_*_gtFine_labelIds.png
    """

    def __init__(
        self,
        root: str,
        split: str = "train",         # "train" | "val" | "test"
        transforms: Optional[A.Compose] = None,
        img_h: int = 512,
        img_w: int = 1024,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.img_h = img_h
        self.img_w = img_w

        # Collect all (image_path, mask_path) pairs
        self.samples: List[Tuple[Path, Path]] = []
        img_dir  = self.root / "leftImg8bit" / split
        mask_dir = self.root / "gtFine" / split

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                "Run print_download_instructions() for setup help."
            )

        for city in sorted(img_dir.iterdir()):
            for img_path in sorted(city.glob("*_leftImg8bit.png")):
                stem = img_path.stem.replace("_leftImg8bit", "")
                mask_path = (mask_dir / city.name /
                             f"{stem}_gtFine_labelIds.png")
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        print(f"[CityscapesDataset] split={split} | {len(self.samples)} samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image (RGB) and label mask (grayscale)
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        label = np.array(Image.open(mask_path), dtype=np.uint8)

        # Convert raw label IDs to training IDs
        label = convert_label_to_trainid(label)

        # Apply augmentations / transforms
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=label)
            image = augmented["image"]          # (C, H, W) float tensor
            label = augmented["mask"].long()    # (H, W) int64 tensor
        else:
            # Fallback: just tensorise
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()

        return {
            "image": image,
            "label": label,
            "path":  str(img_path),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    root: str,
    img_h: int = 512,
    img_w: int = 1024,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Builds train / val / test DataLoaders for Cityscapes.
    Returns a dict with keys 'train', 'val', 'test'.
    """
    datasets = {
        "train": CityscapesDataset(root, "train",
                                   get_train_transforms(img_h, img_w),
                                   img_h, img_w),
        "val":   CityscapesDataset(root, "val",
                                   get_val_transforms(img_h, img_w),
                                   img_h, img_w),
        "test":  CityscapesDataset(root, "test",
                                   get_val_transforms(img_h, img_w),
                                   img_h, img_w),
    }

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if split == "train" else 1,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        print(f"[DataLoader] {split}: {len(ds)} samples, "
              f"batch_size={batch_size if split=='train' else 1}")

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_download_instructions()
    # Uncomment after dataset is downloaded:
    # loaders = get_dataloaders("/data/cityscapes")
    # batch = next(iter(loaders["train"]))
    # print("Image shape:", batch["image"].shape)
    # print("Label shape:", batch["label"].shape)
    # print("Unique label IDs:", batch["label"].unique())
