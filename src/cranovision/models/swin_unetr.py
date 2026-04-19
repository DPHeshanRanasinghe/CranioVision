"""
CranioVision — 3D SwinUNETR model factory.

SwinUNETR (Hatamizadeh et al. 2022) replaces the U-Net's CNN encoder with a
Swin Transformer. The transformer self-attention captures GLOBAL context —
meaning the model "sees" the whole brain at once instead of through local
3×3×3 windows. This consistently puts SwinUNETR at the top of BraTS
leaderboards.

Why SwinUNETR for CranioVision?
  - Complementary to Attention U-Net: CNN biases (local, smooth, high-recall
    for diffuse edema) vs Transformer biases (global, discriminative, better
    for small enhancing cores).
  - Strong XAI hooks: Swin attention maps + Grad-CAM both work.
  - MONAI ships a reference implementation with optional pre-trained weights.

Memory note for Kaggle T4 (16GB):
  feature_size=48 (default) needs ~10-12GB during training at patch 128³.
  If OOM happens, drop feature_size to 24 or 36.
"""
from __future__ import annotations

from typing import Tuple

import torch
from monai.networks.nets import SwinUNETR

from ..config import NUM_CHANNELS, NUM_CLASSES, PATCH_SIZE


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_swin_unetr(
    in_channels: int = NUM_CHANNELS,
    out_channels: int = NUM_CLASSES,
    feature_size: int = 48,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.1,
    use_checkpoint: bool = False,
) -> SwinUNETR:
    """
    Build a 3D SwinUNETR with CranioVision defaults.
    """
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
    )
    return model

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: load MONAI pre-trained weights
# ══════════════════════════════════════════════════════════════════════════════

def load_pretrained_weights(model: SwinUNETR, checkpoint_path: str = None):
    """
    Load MONAI's self-supervised pre-trained SwinUNETR weights.
    This gives a big head-start on small datasets.

    Download the checkpoint from:
      https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV

    File: model_swinvit.pt (about 200MB)

    If checkpoint_path is None, the function is a no-op and prints a hint.
    """
    if checkpoint_path is None:
        print("No pre-trained checkpoint provided — training from scratch.")
        return model

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # The pre-trained checkpoint contains just the Swin encoder
    if "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded pre-trained weights from {checkpoint_path}")
    print(f"  Missing keys   : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
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
    # Run: python -m src.cranovision.models.swin_unetr
    from ..config import DEVICE

    print("=" * 60)
    print("CranioVision — swin_unetr.py smoke test")
    print("=" * 60)

    # Use feature_size=24 for smoke test (fits in 4GB)
    # Real training will use feature_size=48 on Kaggle T4
    model = build_swin_unetr(feature_size=24).to(DEVICE)
    info = count_parameters(model)

    print(f"Device        : {DEVICE}")
    print(f"Feature size  : 24 (smoke test) | 48 (production)")
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

    print("\n✅ swin_unetr.py works.")
    print("\nNote: Production run on Kaggle will use feature_size=48")
    print("      (~62M params, needs ~10GB VRAM during training).")


