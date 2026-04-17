"""
CranioVision — 3D Attention U-Net model factory.

Wraps MONAI's AttentionUnet with CranioVision's defaults:
  - 4 input channels (T1, T1c, T2, FLAIR)
  - 4 output classes (background, edema, enhancing, necrotic)
  - 5-level encoder (channels 32 → 512)
  - Dropout for regularization + future MC Dropout support

Why Attention U-Net?
  The Attention Gates in this architecture learn to suppress irrelevant
  features (skull, healthy tissue) and emphasize salient ones (tumor
  regions). This gives us both better segmentation AND a natural XAI
  hook — we can visualize the attention coefficients directly.
"""
from __future__ import annotations

from typing import Tuple

import torch
from monai.networks.nets import AttentionUnet

from ..config import NUM_CHANNELS, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_attention_unet(
    in_channels: int = NUM_CHANNELS,
    out_channels: int = NUM_CLASSES,
    channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    dropout: float = 0.1,
) -> AttentionUnet:
    """
    Build a 3D Attention U-Net with CranioVision defaults.

    Args:
        in_channels : input modalities (4 for BraTS)
        out_channels: segmentation classes (4: bg + 3 tumor subregions)
        channels    : feature maps per encoder level
                      (32,64,128,256,512 = 5 levels, 4 downsamples)
        strides     : downsampling strides between levels
        dropout     : dropout probability (also used by MC Dropout later)

    Returns:
        A torch.nn.Module ready for training.
    """
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        kernel_size=3,
        up_kernel_size=3,
        dropout=dropout,
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: model info
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: torch.nn.Module) -> dict:
    """Return parameter counts (total and trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run: python -m src.cranovision.models.attention_unet
    from ..config import PATCH_SIZE, DEVICE

    print("=" * 60)
    print("CranioVision — attention_unet.py smoke test")
    print("=" * 60)

    model = build_attention_unet().to(DEVICE)
    info = count_parameters(model)

    print(f"Device        : {DEVICE}")
    print(f"Total params  : {info['total']:>12,}  ({info['total_M']:.2f}M)")
    print(f"Trainable     : {info['trainable']:>12,}  ({info['trainable_M']:.2f}M)")

    # Shape check
    dummy = torch.randn(1, NUM_CHANNELS, *PATCH_SIZE).to(DEVICE)
    with torch.no_grad():
        out = model(dummy)

    print(f"\nShape check:")
    print(f"  Input  : {tuple(dummy.shape)}  (B, C, D, H, W)")
    print(f"  Output : {tuple(out.shape)}    (B, num_classes, D, H, W)")

    assert out.shape == (1, NUM_CLASSES, *PATCH_SIZE), \
        f"Expected (1, {NUM_CLASSES}, {PATCH_SIZE}), got {out.shape}"

    print("\n✅ attention_unet.py works.")