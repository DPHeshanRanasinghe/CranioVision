"""MONAI transform builders for BraTS-style 4-modality segmentation."""

from __future__ import annotations

from typing import Any, Sequence

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
    Transform,
)

IMAGE_KEY = "image"
LABEL_KEY = "label"


def _as_3d_tuple(values: Sequence[int | float], *, field_name: str) -> tuple[int | float, int | float, int | float]:
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values, received {list(values)}")
    return values[0], values[1], values[2]


def _build_shared_transforms(config: dict[str, Any]) -> list[Transform]:
    """Create the deterministic preprocessing steps shared by train/validation."""

    data_cfg = config["data"]
    labels_cfg = config["labels"]
    preprocessing_cfg = config["preprocessing"]

    if len(data_cfg["expected_modalities"]) != 4:
        raise ValueError(
            "Transform pipelines expect 4 input modalities. "
            f"Received {len(data_cfg['expected_modalities'])} configured channels."
        )

    transforms: list[Transform] = [
        # Load four MRI modalities as a stacked image tensor and the segmentation mask separately.
        LoadImaged(keys=[IMAGE_KEY, LABEL_KEY]),
        # Enforce [C, D, H, W] layout for both image and mask before spatial processing.
        EnsureChannelFirstd(keys=[IMAGE_KEY, LABEL_KEY]),
        # Align case orientation before any spacing/cropping so downstream geometry is consistent.
        Orientationd(keys=[IMAGE_KEY, LABEL_KEY], axcodes=preprocessing_cfg["orientation"]),
        # Remap sparse BraTS labels (for example 0/2/3/4) to contiguous training IDs.
        MapLabelValued(
            keys=[LABEL_KEY],
            orig_labels=labels_cfg["original_values"],
            target_labels=labels_cfg["mapped_values"],
        ),
    ]

    if preprocessing_cfg.get("resample", True):
        transforms.append(
            Spacingd(
                keys=[IMAGE_KEY, LABEL_KEY],
                pixdim=_as_3d_tuple(
                    preprocessing_cfg["target_spacing"],
                    field_name="preprocessing.target_spacing",
                ),
                mode=("bilinear", "nearest"),
            )
        )

    transforms.extend(
        [
            # Crop away large zero-valued margins while keeping the image/mask aligned.
            CropForegroundd(keys=[IMAGE_KEY, LABEL_KEY], source_key=IMAGE_KEY),
            # Normalize each MRI channel independently and ignore the zero background.
            NormalizeIntensityd(
                keys=[IMAGE_KEY],
                nonzero=preprocessing_cfg.get("normalize_nonzero", True),
                channel_wise=True,
            ),
        ]
    )
    return transforms


def build_train_transforms(config: dict[str, Any]) -> Compose:
    """Create the stochastic MONAI pipeline used during training."""

    preprocessing_cfg = config["preprocessing"]
    transforms = _build_shared_transforms(config)
    transforms.extend(
        [
            # Sample fixed-size foreground/background-balanced patches for patch-based training.
            RandCropByPosNegLabeld(
                keys=[IMAGE_KEY, LABEL_KEY],
                label_key=LABEL_KEY,
                spatial_size=_as_3d_tuple(
                    preprocessing_cfg["roi_size"],
                    field_name="preprocessing.roi_size",
                ),
                pos=preprocessing_cfg["crop_pos"],
                neg=preprocessing_cfg["crop_neg"],
                num_samples=preprocessing_cfg["train_samples_per_volume"],
                image_key=IMAGE_KEY,
                image_threshold=0.0,
            ),
            RandFlipd(keys=[IMAGE_KEY, LABEL_KEY], prob=0.5, spatial_axis=0),
            RandFlipd(keys=[IMAGE_KEY, LABEL_KEY], prob=0.5, spatial_axis=1),
            RandFlipd(keys=[IMAGE_KEY, LABEL_KEY], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=[IMAGE_KEY], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=[IMAGE_KEY], offsets=0.1, prob=0.5),
            EnsureTyped(keys=[IMAGE_KEY, LABEL_KEY]),
        ]
    )
    return Compose(transforms)


def build_val_transforms(config: dict[str, Any]) -> Compose:
    """Create the deterministic MONAI pipeline used for validation and inference."""

    transforms = _build_shared_transforms(config)
    transforms.append(EnsureTyped(keys=[IMAGE_KEY, LABEL_KEY]))
    return Compose(transforms)
