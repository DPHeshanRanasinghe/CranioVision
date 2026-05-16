"""
CranioVision — Segmentation metrics.

Thin wrappers around MONAI's DiceMetric and HausdorffDistanceMetric,
preconfigured for BraTS-style multi-class 3D segmentation.

Also provides BraTS-standard region metrics:
  - WT (Whole Tumor)     : labels 1 + 2 + 3 (edema + enhancing + necrotic)
  - TC (Tumor Core)      : labels 2 + 3 (enhancing + necrotic)
  - ET (Enhancing Tumor) : label 2

These regions are the official BraTS challenge evaluation targets.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose

from ..config import NUM_CLASSES, CLASS_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING — convert raw logits to class predictions for metrics
# ══════════════════════════════════════════════════════════════════════════════

post_pred  = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# CORE METRIC FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def make_dice_metric() -> DiceMetric:
    """
    Dice per foreground class (excludes background).
    reduction='mean_batch' averages over batch dim and keeps class dim,
    so aggregate() returns a tensor of shape (num_foreground_classes,).
    """
    return DiceMetric(
        include_background=False,
        reduction="mean_batch",
        get_not_nans=False,
    )


def make_hd95_metric() -> HausdorffDistanceMetric:
    """
    Hausdorff distance at the 95th percentile — measures boundary accuracy.
    Reported in voxel units (1mm in BraTS isotropic data).
    """
    return HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean_batch",
        get_not_nans=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PER-CASE DICE (for per-patient reporting)
# ══════════════════════════════════════════════════════════════════════════════

def compute_case_dice(pred_mask: torch.Tensor,
                      true_mask: torch.Tensor,
                      num_classes: int = NUM_CLASSES,
                      include_background: bool = False) -> List[float]:
    """
    Compute Dice for ONE case across each class.
    Returns list of floats — one per class (excluding background by default).
    pred_mask and true_mask should be integer tensors of identical shape.
    """
    start = 0 if include_background else 1
    dices = []
    for c in range(start, num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        intersection = (pred_c * true_c).sum()
        denom = pred_c.sum() + true_c.sum()
        if denom.item() == 0:
            # Class not present in either — convention: perfect (1.0)
            dices.append(1.0)
        else:
            dices.append((2.0 * intersection / denom).item())
    return dices


# ══════════════════════════════════════════════════════════════════════════════
# BRATS REGION METRICS (WT / TC / ET)
# ══════════════════════════════════════════════════════════════════════════════

# Internal label mapping (after MapLabelValued):
#   0 = background
#   1 = edema
#   2 = enhancing tumor
#   3 = necrotic core
BRATS_REGIONS: Dict[str, List[int]] = {
    "WT": [1, 2, 3],   # Whole tumor  = edema + enhancing + necrotic
    "TC": [2, 3],      # Tumor core   = enhancing + necrotic
    "ET": [2],         # Enhancing tumor only
}


def build_region_mask(mask: torch.Tensor, labels: List[int]) -> torch.Tensor:
    """Binary mask = True wherever input belongs to any of the given labels."""
    out = torch.zeros_like(mask, dtype=torch.float32)
    for lbl in labels:
        out = out + (mask == lbl).float()
    return (out > 0).float()


def compute_brats_region_dice(pred_mask: torch.Tensor,
                              true_mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute Dice for each BraTS region (WT, TC, ET) for ONE case.
    pred_mask and true_mask: integer tensors, identical shape.
    """
    results = {}
    for region, labels in BRATS_REGIONS.items():
        pred_r = build_region_mask(pred_mask, labels)
        true_r = build_region_mask(true_mask, labels)
        intersection = (pred_r * true_r).sum()
        denom = pred_r.sum() + true_r.sum()
        if denom.item() == 0:
            results[region] = 1.0
        else:
            results[region] = (2.0 * intersection / denom).item()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def format_per_class_dice(dice_tensor: torch.Tensor) -> str:
    """Format aggregated DiceMetric output as human-readable string."""
    # dice_tensor has 3 values: edema, enhancing, necrotic (background excluded)
    names = [n for n in CLASS_NAMES[1:]]
    parts = []
    for name, value in zip(names, dice_tensor.tolist()):
        short = name.split()[0][:3]   # "Ede", "Enh", "Nec"
        parts.append(f"{short}:{value:.3f}")
    return " ".join(parts)


def format_region_dice(regions: Dict[str, float]) -> str:
    """Format BraTS region dict as human-readable string."""
    return " ".join(f"{k}:{v:.3f}" for k, v in regions.items())


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run: python -m src.cranovision.training.metrics
    print("=" * 60)
    print("CranioVision — metrics.py smoke test")
    print("=" * 60)

    # Create a synthetic prediction and ground truth
    torch.manual_seed(0)
    shape = (1, 1, 32, 32, 32)   # (B, 1, H, W, D)
    gt   = torch.randint(0, 4, shape)
    pred = gt.clone()
    # Corrupt 10% of voxels to make Dice < 1
    noise = torch.randint(0, 4, shape)
    mask  = torch.rand(shape) < 0.10
    pred[mask] = noise[mask]

    print("\n─── Per-class Dice ───")
    per_class = compute_case_dice(pred, gt)
    for name, d in zip(CLASS_NAMES[1:], per_class):
        print(f"  {name:18s}: {d:.4f}")

    print("\n─── BraTS region Dice ───")
    regions = compute_brats_region_dice(pred, gt)
    for region, d in regions.items():
        print(f"  {region}: {d:.4f}")

    print("\n─── MONAI Dice metric object ───")
    metric = make_dice_metric()
    print(f"  include_background: {metric.include_background}")
    print(f"  reduction         : {metric.reduction}")

    print("\n─── MONAI HD95 metric object ───")
    hd95 = make_hd95_metric()
    print(f"  percentile        : {hd95.percentile}")

    print("\n✅ metrics.py works.")