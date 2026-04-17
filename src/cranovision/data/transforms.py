"""
CranioVision — MONAI preprocessing transforms.

Provides three Compose pipelines:
  - get_train_transforms() : with augmentation, random patches
  - get_val_transforms()   : deterministic, center crop
  - get_test_transforms()  : same as val, used at inference time

All three share the same base preprocessing (load, orient, normalize, crop)
so the model sees consistent input distributions.
"""
from __future__ import annotations

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    CenterSpatialCropd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapLabelValued,
)

from ..config import (
    PATCH_SIZE,
    VAL_SPATIAL_SIZE,
    ORIENTATION_AXCODES,
    LABEL_MAP,
)


# ══════════════════════════════════════════════════════════════════════════════
# BASE TRANSFORMS (shared by train/val/test)
# ══════════════════════════════════════════════════════════════════════════════

def _base_transforms():
    """
    Deterministic preprocessing applied to all splits.

    Order matters:
    1. Load all 4 modality files + seg mask
    2. Add channel dim (images become 4-channel, label 1-channel)
    3. Remap BraTS labels {0,2,3,4} -> {0,1,2,3}
    4. Reorient to RAS standard
    5. Z-score normalize per modality (ignoring background zeros)
    6. Crop to brain bounding box (remove zero borders)
    7. Force float32 image / long label
    """
    return [
        LoadImaged(keys=["image", "label"]),

        EnsureChannelFirstd(keys=["image", "label"]),

        # BraTS 2024 uses labels 0/2/3/4 — remap to 0/1/2/3
        MapLabelValued(
            keys=["label"],
            orig_labels=list(LABEL_MAP.keys()),
            target_labels=list(LABEL_MAP.values()),
        ),

        Orientationd(keys=["image", "label"], axcodes=ORIENTATION_AXCODES),

        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,        # only compute mean/std on non-zero voxels
            channel_wise=True,   # each modality normalized independently
        ),

        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            allow_smaller=True,
        ),

        EnsureTyped(
            keys=["image", "label"],
            dtype=[torch.float32, torch.long],
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING TRANSFORMS — with augmentation
# ══════════════════════════════════════════════════════════════════════════════

def get_train_transforms(patch_size=PATCH_SIZE) -> Compose:
    """
    Training pipeline: base + random crop + augmentation.

    Random crop picks a 128^3 sub-volume each iteration — so a single
    patient can contribute many different training samples across epochs.
    """
    train_aug = [
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=patch_size,
            random_size=False,
        ),

        # Random axis flips — brain has approximate left-right symmetry
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

        # 90-degree rotations in axial plane
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

        # Intensity augmentation — mimics scanner variability
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ]
    return Compose(_base_transforms() + train_aug)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION TRANSFORMS — deterministic
# ══════════════════════════════════════════════════════════════════════════════

def get_val_transforms(spatial_size=VAL_SPATIAL_SIZE) -> Compose:
    """
    Validation pipeline: base + center crop.
    No augmentation — we want deterministic metric computation.
    """
    val_crop = [
        CenterSpatialCropd(
            keys=["image", "label"],
            roi_size=spatial_size,
        ),
    ]
    return Compose(_base_transforms() + val_crop)


# ══════════════════════════════════════════════════════════════════════════════
# TEST TRANSFORMS — same as val
# ══════════════════════════════════════════════════════════════════════════════

def get_test_transforms(spatial_size=VAL_SPATIAL_SIZE) -> Compose:
    """
    Test-time pipeline (also used for inference).
    Same as validation — deterministic, no augmentation.
    At inference with a sliding-window inferer you can skip the center crop
    entirely; this version still crops for memory safety on small GPUs.
    """
    return get_val_transforms(spatial_size)


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run: python -m src.cranovision.data.transforms
    import numpy as np
    from .dataset import get_splits

    print("=" * 60)
    print("CranioVision — transforms.py smoke test")
    print("=" * 60)

    train, _, _ = get_splits(verbose=False)
    print(f"Loaded {len(train)} train cases")

    print("\nBuilding train transforms...")
    train_t = get_train_transforms()
    print("Running one training sample through pipeline (takes ~20s)...")

    sample = train_t(train[0])
    img, lbl = sample["image"], sample["label"]

    print("\n─── Sanity checks ───")
    print(f"Image shape   : {tuple(img.shape)}    (expect: (4, 128, 128, 128))")
    print(f"Label shape   : {tuple(lbl.shape)}    (expect: (1, 128, 128, 128))")
    print(f"Image dtype   : {img.dtype}")
    print(f"Label dtype   : {lbl.dtype}")
    print(f"Image range   : [{img.min():.3f}, {img.max():.3f}]  (z-scored, centred near 0)")
    print(f"Label unique  : {sorted(lbl.unique().tolist())}  (expect: subset of [0,1,2,3])")

    assert img.shape[0] == 4, "Image should have 4 channels (4 modalities)"
    assert img.ndim == 4,     "Image should be 4D: (C, H, W, D)"
    assert lbl.ndim == 4,     "Label should be 4D: (1, H, W, D)"
    assert set(lbl.unique().tolist()).issubset({0, 1, 2, 3}), \
        "Label values should be in {0,1,2,3} after remapping"

    print("\n✅ transforms.py works.")