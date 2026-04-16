"""Config-driven SwinUNETR builder."""

from __future__ import annotations

from typing import Any, Mapping

from monai.networks.nets import SwinUNETR


def build_swin_unetr(config: Mapping[str, Any]) -> SwinUNETR:
    """Build a 3D MONAI SwinUNETR from the project config."""

    model_cfg = config["model"] if "model" in config else config
    preprocessing_cfg = config.get("preprocessing", {})

    roi_size = model_cfg.get("img_size", preprocessing_cfg.get("roi_size"))
    if roi_size is None:
        raise ValueError(
            "SwinUNETR requires model.img_size or preprocessing.roi_size in the config."
        )

    img_size = tuple(int(value) for value in roi_size)
    if len(img_size) != 3:
        raise ValueError(f"SwinUNETR expects a 3D img_size, received {img_size}.")

    feature_size = int(model_cfg.get("feature_size", 48))
    if feature_size % 12 != 0:
        raise ValueError(
            "SwinUNETR feature_size must be divisible by 12. "
            f"Received feature_size={feature_size}."
        )

    return SwinUNETR(
        img_size=img_size,
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        feature_size=feature_size,
        use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
    )
