"""MONAI transform builders for BraTS-style segmentation."""

from __future__ import annotations

from typing import Any

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)


def _common_transforms(config: dict[str, Any]) -> list[Any]:
    label_cfg = config["labels"]
    preprocess_cfg = config["preprocessing"]

    transforms: list[Any] = [
        # Load image and label volumes from NIfTI paths stored in the dataset dictionary.
        LoadImaged(keys=["image", "label"]),
        # Guarantee channel-first tensors so MONAI models consistently receive [C, D, H, W].
        EnsureChannelFirstd(keys=["image", "label"]),
        # Standardize orientation across cases before any spatial processing.
        Orientationd(keys=["image", "label"], axcodes=preprocess_cfg["orientation"]),
        # BraTS masks often use labels {0, 2, 3, 4}; remap them to contiguous IDs for training.
        MapLabelValued(
            keys=["label"],
            orig_labels=label_cfg["original_values"],
            target_labels=label_cfg["mapped_values"],
        ),
    ]

    if preprocess_cfg.get("resample", True):
        transforms.append(
            # Resample all cases to a common voxel spacing so patch sizes are comparable.
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(preprocess_cfg["target_spacing"]),
                mode=("bilinear", "nearest"),
            )
        )

    transforms.extend(
        [
            # Remove large zero-background margins to reduce wasted computation on empty space.
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # Normalize each MRI channel independently while ignoring the zero background.
            NormalizeIntensityd(
                keys=["image"],
                nonzero=preprocess_cfg.get("normalize_nonzero", True),
                channel_wise=True,
            ),
        ]
    )

    return transforms


def build_train_transforms(config: dict[str, Any]) -> Compose:
    """Create the stochastic MONAI pipeline used during training."""

    preprocess_cfg = config["preprocessing"]
    transforms = _common_transforms(config)
    transforms.extend(
        [
            # Sample fixed-size 3D patches and balance tumor/background regions.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(preprocess_cfg["roi_size"]),
                pos=preprocess_cfg["crop_pos"],
                neg=preprocess_cfg["crop_neg"],
                num_samples=preprocess_cfg["train_samples_per_volume"],
                image_key="image",
                image_threshold=0.0,
            ),
            # Basic geometric augmentation for 3D robustness without changing anatomy semantics.
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # Mild intensity perturbations help the model generalize across scanners and sites.
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return Compose(transforms)


def build_val_transforms(config: dict[str, Any]) -> Compose:
    """Create the deterministic MONAI pipeline used for validation and inference."""

    transforms = _common_transforms(config)
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)
