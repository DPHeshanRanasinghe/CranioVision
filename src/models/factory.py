"""Simple segmentation model factory."""

from __future__ import annotations

from typing import Any

from monai.networks.nets import AttentionUnet, SwinUNETR, UNet


def build_model(config: dict[str, Any]):
    """Build the segmentation model selected in the YAML config."""

    model_cfg = config["model"]
    model_name = model_cfg["name"].lower()

    if model_name in {"unet", "monai_unet"}:
        return UNet(
            spatial_dims=3,
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            channels=tuple(model_cfg["channels"]),
            strides=tuple(model_cfg["strides"]),
            num_res_units=model_cfg.get("num_res_units", 2),
        )

    if model_name == "attention_unet":
        return AttentionUnet(
            spatial_dims=3,
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            channels=tuple(model_cfg["channels"]),
            strides=tuple(model_cfg["strides"]),
        )

    if model_name == "swinunetr":
        return SwinUNETR(
            img_size=tuple(model_cfg.get("img_size", config["preprocessing"]["roi_size"])),
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            feature_size=model_cfg.get("feature_size", 48),
            use_checkpoint=model_cfg.get("use_checkpoint", False),
        )

    raise ValueError(
        f"Unsupported model '{model_cfg['name']}'. "
        "Expected one of: unet, attention_unet, swinunetr."
    )
