"""
CranioVision — MONAI preprocessing transforms.
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
    SpatialPadd,
)

from ..config import (
    PATCH_SIZE,
    VAL_SPATIAL_SIZE,
    ORIENTATION_AXCODES,
    LABEL_MAP,
)


def _base_transforms(min_spatial_size):
    """
    Deterministic preprocessing applied to all splits.
    min_spatial_size ensures the volume is large enough for the crop stage.

    ``allow_missing_keys=True`` on every label-touching op so production
    uploads (no GT) are processed cleanly — the transforms simply no-op
    on the label key when the case_dict doesn't include one.
    """
    return [
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        MapLabelValued(
            keys=["label"],
            orig_labels=list(LABEL_MAP.keys()),
            target_labels=list(LABEL_MAP.values()),
            allow_missing_keys=True,
        ),
        Orientationd(
            keys=["image", "label"], axcodes=ORIENTATION_AXCODES,
            allow_missing_keys=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "label"], source_key="image",
            allow_smaller=True, allow_missing_keys=True,
        ),
        # Ensure we never end up smaller than the required crop size
        SpatialPadd(
            keys=["image", "label"], spatial_size=min_spatial_size,
            mode="constant", allow_missing_keys=True,
        ),
        EnsureTyped(
            keys=["image", "label"], dtype=[torch.float32, torch.long],
            allow_missing_keys=True,
        ),
    ]


def get_train_transforms(patch_size=PATCH_SIZE) -> Compose:
    """Training pipeline: base + random crop + augmentation."""
    train_aug = [
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=patch_size,
            random_size=False,
            allow_missing_keys=True,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0, allow_missing_keys=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2, allow_missing_keys=True),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, allow_missing_keys=True),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ]
    return Compose(_base_transforms(min_spatial_size=patch_size) + train_aug)


def get_val_transforms(spatial_size=VAL_SPATIAL_SIZE) -> Compose:
    """Validation pipeline: base + center crop. No augmentation."""
    val_crop = [
        CenterSpatialCropd(
            keys=["image", "label"], roi_size=spatial_size,
            allow_missing_keys=True,
        ),
    ]
    return Compose(_base_transforms(min_spatial_size=spatial_size) + val_crop)


def get_test_transforms(spatial_size=VAL_SPATIAL_SIZE) -> Compose:
    return get_val_transforms(spatial_size)


if __name__ == "__main__":
    import numpy as np
    from .dataset import get_splits

    print("=" * 60)
    print("CranioVision — transforms.py smoke test")
    print("=" * 60)

    train, _, _ = get_splits(verbose=False)
    print(f"Loaded {len(train)} train cases")

    train_t = get_train_transforms()
    sample = train_t(train[0])
    img, lbl = sample["image"], sample["label"]

    print(f"Image shape   : {tuple(img.shape)}    (expect: (4, 128, 128, 128))")
    print(f"Label shape   : {tuple(lbl.shape)}    (expect: (1, 128, 128, 128))")
    print(f"Label unique  : {sorted(lbl.unique().tolist())}")

    assert img.shape[0] == 4
    assert img.shape[1:] == (128, 128, 128), f"Training output must be exactly 128³, got {img.shape[1:]}"
    print("\n✅ transforms.py works.")