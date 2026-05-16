"""
CranioVision — Inference module.

Loads a trained checkpoint and runs sliding-window inference on one case.
Works for all 3 models (Attention U-Net, SwinUNETR, nnU-Net) because they
all produce the same output shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer

from ..config import DEVICE, PATCH_SIZE, USE_AMP, NUM_CLASSES, CLASS_NAMES
from ..data import get_val_transforms
from ..models import (
    build_attention_unet,
    build_swin_unetr,
    build_nnunet_for_training,
)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_state_dict(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """Load .pth state dict with graceful handling of DataParallel / wrappers."""
    # Always deserialize checkpoints on CPU first. Loading directly to CUDA can
    # briefly duplicate model weights on the GPU and OOM during ensemble demos.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # If saved from DataParallel, keys start with 'module.'
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠ Missing keys: {len(missing)}")
    if unexpected:
        print(f"  ⚠ Unexpected keys: {len(unexpected)}")
    return model


def load_model(model_name: str, ckpt_path: Path, **build_kwargs) -> torch.nn.Module:
    """
    Factory: build the right architecture and load the checkpoint.

    Args:
        model_name: 'attention_unet' | 'swin_unetr' | 'nnunet'
        ckpt_path : path to the .pth file
        build_kwargs: forwarded to the model builder

    Returns:
        A ready-to-use model in eval() mode on DEVICE.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {model_name} from {ckpt_path.name}...")

    name = model_name.lower()
    if name in ("attention_unet", "attention", "attn"):
        model = build_attention_unet(**build_kwargs)
    elif name in ("swin_unetr", "swin"):
        # Must match training config
        defaults = {"feature_size": 48, "use_checkpoint": False}
        defaults.update(build_kwargs)
        model = build_swin_unetr(**defaults)
    elif name in ("nnunet", "nn_unet", "dynunet"):
        defaults = {
            "filters": (32, 64, 128, 256, 320, 320),
            "deep_supervision": True,
            "deep_supr_num": 2,
        }
        defaults.update(build_kwargs)
        model = build_nnunet_for_training(**defaults)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    _load_state_dict(model, ckpt_path)
    model = model.to(DEVICE).eval()
    print(f"  ✓ Loaded. Device: {DEVICE}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-CASE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def make_inferer(patch_size: Tuple[int, ...] = PATCH_SIZE,
                 sw_batch_size: int = 2,
                 overlap: float = 0.5) -> SlidingWindowInferer:
    """Standard sliding-window inferer — same as used during validation."""
    return SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
    )


def preprocess_case(case_dict: Dict) -> Dict:
    """
    Apply the standard val transforms to one case dict.
    case_dict format: {'image': [4 paths], 'label': path, 'case_id': str}
    Returns the transformed dict — ready for inference.
    """
    transforms = get_val_transforms()
    return transforms(case_dict)


def predict_case(
    model: torch.nn.Module,
    case_dict: Dict,
    inferer: Optional[SlidingWindowInferer] = None,
    return_probabilities: bool = False,
) -> Dict:
    """
    Run inference on a single case.

    Returns a dict with:
      'image'   : preprocessed input tensor (4, H, W, D)
      'label'   : ground-truth mask (1, H, W, D) — if available
      'pred'    : predicted class mask (H, W, D) as int64
      'probs'   : (optional) softmax probabilities (4, H, W, D) — if return_probabilities=True
      'case_id' : patient identifier
    """
    if inferer is None:
        inferer = make_inferer()

    # Preprocess
    sample = preprocess_case(case_dict)
    image = sample["image"].unsqueeze(0).to(DEVICE)    # add batch dim -> (1, 4, H, W, D)

    # Forward
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = inferer(image, model)             # (1, 4, H, W, D)

    probs = torch.softmax(logits, dim=1)               # (1, 4, H, W, D)
    pred  = probs.argmax(dim=1).squeeze(0)             # (H, W, D) int

    result = {
        "image"  : sample["image"].cpu(),
        "label"  : sample["label"].cpu() if "label" in sample else None,
        "pred"   : pred.cpu(),
        "case_id": case_dict.get("case_id", "unknown"),
    }
    if return_probabilities:
        result["probs"] = probs.squeeze(0).cpu()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_region_volumes(mask: torch.Tensor,
                           voxel_volume_mm3: float = 1.0) -> Dict[str, float]:
    """
    Compute tumor volume (in cm³) per class from a prediction mask.

    Args:
        mask: int tensor, shape (H, W, D) — class indices 0..3
        voxel_volume_mm3: spatial volume of one voxel (1.0 for BraTS isotropic)

    Returns dict:
      {
        'Edema'           : cm³,
        'Enhancing tumor' : cm³,
        'Necrotic core'   : cm³,
        'Total tumor'     : cm³,
      }
    """
    if mask.ndim == 4:
        mask = mask.squeeze(0)

    out = {}
    total = 0.0
    for class_idx in range(1, NUM_CLASSES):   # skip background
        n_voxels = (mask == class_idx).sum().item()
        volume_cm3 = n_voxels * voxel_volume_mm3 / 1000.0
        name = CLASS_NAMES[class_idx]
        out[name] = round(volume_cm3, 2)
        total += volume_cm3

    out["Total tumor"] = round(total, 2)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run: python -m src.cranovision.inference.predict
    from ..config import MODELS_DIR
    from ..data import get_splits

    print("=" * 60)
    print("CranioVision — predict.py smoke test")
    print("=" * 60)

    ckpt = MODELS_DIR / "attention_unet_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {ckpt} — download from Kaggle first."
        )

    model = load_model("attention_unet", ckpt)

    _, _, test_cases = get_splits(verbose=False)
    print(f"\nRunning inference on first test case...")
    result = predict_case(model, test_cases[0])

    print(f"\n─── Result ───")
    print(f"Case ID    : {result['case_id']}")
    print(f"Pred shape : {tuple(result['pred'].shape)}")
    print(f"Pred unique: {sorted(result['pred'].unique().tolist())}")

    volumes = compute_region_volumes(result['pred'])
    print(f"\n─── Volumes ───")
    for name, vol in volumes.items():
        print(f"  {name:20s}: {vol:>7.2f} cm³")

    print("\n✅ predict.py works.")
