"""Config-driven Attention U-Net builder."""

from __future__ import annotations

from typing import Any, Mapping

from monai.networks.nets import AttentionUnet


def build_attention_unet(config: Mapping[str, Any]) -> AttentionUnet:
    """Build a 3D MONAI Attention U-Net from the project config."""

    model_cfg = config["model"] if "model" in config else config
    channels = tuple(int(channel) for channel in model_cfg["channels"])
    strides = tuple(int(stride) for stride in model_cfg["strides"])

    if len(channels) != len(strides) + 1:
        raise ValueError(
            "Attention U-Net expects len(channels) == len(strides) + 1. "
            f"Received channels={channels} and strides={strides}."
        )

    return AttentionUnet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        channels=channels,
        strides=strides,
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
