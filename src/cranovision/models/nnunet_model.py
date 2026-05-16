"""
CranioVision — nnU-Net-style architecture (Option B integration).

Uses MONAI's DynUNet with hyperparameters matching nnU-Net's auto-configured
defaults for brain MRI segmentation. This gives us nnU-Net's architectural
strengths while keeping our unified training pipeline.

Why DynUNet?
  DynUNet is MONAI's direct implementation of the "Dynamic U-Net" from
  nnU-Net's paper. Same building blocks, same topology, same residual
  connections. The only thing we skip is nnU-Net's auto-plan generator —
  we set the plan manually using known-good values for BraTS-like data.

Hyperparameters chosen from nnU-Net's published plans for BraTS:
  - Kernel sizes : 3x3x3 at all levels
  - Strides      : 1 at level 0, then 2 at each downsampling step
  - Filters      : (32, 64, 128, 256, 320, 320) — nnU-Net caps at 320
  - Deep supervision: outputs from intermediate decoder levels supervised
                       during training (improves gradient flow)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

from ..config import NUM_CHANNELS, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY — nnU-Net-style DynUNet
# ══════════════════════════════════════════════════════════════════════════════

def build_nnunet(
    in_channels: int = NUM_CHANNELS,
    out_channels: int = NUM_CLASSES,
    filters: Tuple[int, ...] = (32, 64, 128, 256, 320, 320),
    deep_supervision: bool = True,
    deep_supr_num: int = 2,
    dropout: float = 0.1,
) -> DynUNet:
    """
    Build a nnU-Net-style 3D DynUNet configured for BraTS-like data.

    Args:
        in_channels     : input modalities (4 for BraTS)
        out_channels    : segmentation classes (4: bg + 3 tumor subregions)
        filters         : feature maps per encoder level (nnU-Net caps at 320)
        deep_supervision: emit intermediate decoder outputs for auxiliary loss
        deep_supr_num   : how many auxiliary outputs (top-K deepest levels)
        dropout         : dropout probability in decoder (also enables MC Dropout)

    Returns:
        torch.nn.Module. When deep_supervision=True and model is in train mode,
        forward returns shape (B, num_supervision_heads+1, C, D, H, W).
        In eval mode it returns a single (B, C, D, H, W) tensor.
    """
    # nnU-Net's standard 3D plan for brain MRI:
    # Level 0: stride 1 (keep full res)
    # Levels 1..5: stride 2 (downsample by 2 each time)
    # All kernels are 3x3x3
    n_levels = len(filters)
    strides  = [1] + [2] * (n_levels - 1)
    kernels  = [[3, 3, 3]] * n_levels
    upsample_kernels = strides[1:]   # length must be n_levels - 1

    model = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=upsample_kernels,
        filters=list(filters),
        norm_name="instance",            # instance norm — standard for nnU-Net
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=deep_supervision,
        deep_supr_num=deep_supr_num,
        res_block=True,                  # residual connections in conv blocks
        trans_bias=True,
        dropout=dropout,
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# DEEP SUPERVISION WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class DeepSupWrapper(nn.Module):
    """
    Wraps DynUNet so that during training it returns only the main output —
    compatible with our standard DiceCELoss pipeline.

    The auxiliary deep-supervision outputs CAN be used for a richer loss,
    but that requires a custom training loop. For parity with Attention U-Net
    and SwinUNETR in CranioVision, we take only the highest-resolution head.

    If you want to use the full deep supervision loss later, just unwrap and
    sum individual head losses in your training loop.
    """
    def __init__(self, model: DynUNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # In train mode with deep_supervision, DynUNet returns a stacked tensor
        # along dim=1: shape (B, num_heads, C, D, H, W). Main head is index 0.
        if out.ndim == 6:
            return out[:, 0]          # take main output head
        return out


def build_nnunet_for_training(**kwargs) -> nn.Module:
    """
    Build nnU-Net wrapped for standard (non-deep-sup) training.
    Drop-in replacement for build_attention_unet / build_swin_unetr.
    """
    base = build_nnunet(**kwargs)
    return DeepSupWrapper(base)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> dict:
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
    # Run: python -m src.cranovision.models.nnunet_model
    from ..config import DEVICE, PATCH_SIZE

    print("=" * 60)
    print("CranioVision — nnunet_model.py smoke test")
    print("=" * 60)

    # Smaller filters for smoke test so 4GB GPU survives
    smoke_filters = (16, 32, 64, 128, 160, 160)
    model = build_nnunet_for_training(filters=smoke_filters).to(DEVICE)
    info = count_parameters(model)

    print(f"Device        : {DEVICE}")
    print(f"Filters used  : {smoke_filters} (smoke) | (32,64,128,256,320,320) production")
    print(f"Total params  : {info['total']:>12,}  ({info['total_M']:.2f}M)")
    print(f"Trainable     : {info['trainable']:>12,}  ({info['trainable_M']:.2f}M)")

    # Shape check — TRAIN mode first (deep supervision active)
    model.train()
    dummy = torch.randn(1, NUM_CHANNELS, *PATCH_SIZE).to(DEVICE)
    with torch.no_grad():
        out = model(dummy)
    print(f"\nShape check (TRAIN mode):")
    print(f"  Input  : {tuple(dummy.shape)}  (B, C, D, H, W)")
    print(f"  Output : {tuple(out.shape)}    (B, num_classes, D, H, W)")

    assert out.shape == (1, NUM_CLASSES, *PATCH_SIZE), \
        f"Expected (1, {NUM_CLASSES}, {PATCH_SIZE}), got {out.shape}"

    # Also check EVAL mode
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy)
    print(f"\nShape check (EVAL mode):")
    print(f"  Output : {tuple(out_eval.shape)}")
    assert out_eval.shape == (1, NUM_CLASSES, *PATCH_SIZE)

    print("\n✅ nnunet_model.py works.")
    print("\nNote: Production run on Kaggle will use filters=(32,64,128,256,320,320)")
    print("      (~30M params, needs ~8GB VRAM during training).")