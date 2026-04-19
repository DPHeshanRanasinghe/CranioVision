"""
CranioVision — Monte Carlo Dropout (epistemic uncertainty).

Standard inference gives a single "confident-looking" segmentation — but the
model has no idea how confident it actually is. MC Dropout solves this:

1. Keep dropout layers ACTIVE at inference time (they're normally disabled)
2. Run the same input N times — each pass has different random dropouts,
   so predictions vary slightly
3. Mean across passes = final prediction
4. Standard deviation across passes = uncertainty

High variance voxels = model is unsure → flag for radiologist review.
Low variance voxels = model is confident.

This is the SAFETY NET of the CranioVision pipeline.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer

from ..config import DEVICE, USE_AMP, PATCH_SIZE, NUM_CLASSES, CLASS_NAMES
from ..data import get_val_transforms
from .predict import make_inferer


# ══════════════════════════════════════════════════════════════════════════════
# ENABLE DROPOUT AT INFERENCE TIME
# ══════════════════════════════════════════════════════════════════════════════

def enable_dropout(model: nn.Module) -> None:
    """
    Flip all Dropout layers to training mode while keeping BatchNorm in eval.
    This is the critical trick that makes MC Dropout work.
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def count_dropout_layers(model: nn.Module) -> int:
    """Sanity check — how many Dropout layers does the model have?"""
    return sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
    )


# ══════════════════════════════════════════════════════════════════════════════
# MC DROPOUT INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def mc_dropout_predict(
    model: nn.Module,
    case_dict: Dict,
    n_samples: int = 20,
    inferer: Optional[SlidingWindowInferer] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run N stochastic forward passes with dropout active, then aggregate.

    Returns dict with:
      'mean_probs'    : (C, H, W, D) averaged softmax probabilities
      'std_probs'     : (C, H, W, D) std dev across samples — uncertainty per class
      'pred'          : (H, W, D) final prediction (argmax of mean_probs)
      'uncertainty'   : (H, W, D) per-voxel total uncertainty (mean std across classes)
      'confidence'    : (H, W, D) per-voxel confidence = 1 - uncertainty
      'label'         : (1, H, W, D) ground truth if available
      'case_id'       : str
      'n_samples'     : int

    Runtime: ~N× slower than a single predict. On 4GB GPU with 128³ patches,
    expect ~1-2 min per sample × 20 samples = 20-40 min for one full case.
    For demos, use n_samples=5. For real uncertainty, use 20-30.
    """
    if inferer is None:
        inferer = make_inferer()

    # Preprocess input
    transforms = get_val_transforms()
    sample = transforms(case_dict)
    image = sample["image"].unsqueeze(0).to(DEVICE)

    # Put model in eval mode (freezes BatchNorm) then re-enable dropout
    model.eval()
    enable_dropout(model)
    n_drop = count_dropout_layers(model)
    if n_drop == 0:
        raise RuntimeError(
            "Model has NO Dropout layers — MC Dropout cannot work. "
            "Rebuild the model with dropout > 0."
        )
    if verbose:
        print(f"MC Dropout active — {n_drop} dropout layers enabled")
        print(f"Running {n_samples} stochastic forward passes...")

    # Accumulate probability samples
    # Shape accumulators: use a running mean to save memory
    running_sum    = None     # shape (C, H, W, D)
    running_sumsq  = None     # shape (C, H, W, D) — for variance

    for i in range(n_samples):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = inferer(image, model)        # (1, C, H, W, D)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().float()   # (C, H, W, D)

        if running_sum is None:
            running_sum   = probs.clone()
            running_sumsq = probs.pow(2).clone()
        else:
            running_sum   += probs
            running_sumsq += probs.pow(2)

        if verbose:
            print(f"  Sample {i+1:2d}/{n_samples}")

    # Mean and std across samples (numerically stable variance formula)
    mean_probs = running_sum / n_samples
    # Var = E[X²] - (E[X])²
    var_probs = (running_sumsq / n_samples) - mean_probs.pow(2)
    std_probs = var_probs.clamp(min=0).sqrt()

    # Final prediction
    pred = mean_probs.argmax(dim=0)                   # (H, W, D)

    # Per-voxel uncertainty = mean std across classes (simple, interpretable)
    # Alternative: entropy of mean_probs — we use std for speed
    uncertainty = std_probs.mean(dim=0)               # (H, W, D)
    confidence  = 1.0 - uncertainty.clamp(0, 1)

    # Return model to full eval mode
    model.eval()

    return {
        "mean_probs"  : mean_probs,
        "std_probs"   : std_probs,
        "pred"        : pred,
        "uncertainty" : uncertainty,
        "confidence"  : confidence,
        "label"       : sample["label"].cpu() if "label" in sample else None,
        "image"       : sample["image"].cpu(),
        "case_id"     : case_dict.get("case_id", "unknown"),
        "n_samples"   : n_samples,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════════════

def summarize_confidence(result: Dict,
                         uncertain_threshold: float = 0.15) -> Dict:
    """
    Summarize confidence across the predicted tumor.

    Args:
        result: output of mc_dropout_predict
        uncertain_threshold: voxels with std > this are "uncertain"

    Returns dict:
        'mean_confidence'       : float — average 1-std over tumor voxels
        'uncertain_voxel_count' : int   — how many tumor voxels are uncertain
        'uncertain_fraction'    : float — fraction of tumor that's uncertain
        'per_class'             : per-class confidence summary
    """
    pred = result["pred"]
    unc  = result["uncertainty"]

    tumor_mask = pred > 0
    n_tumor = tumor_mask.sum().item()

    if n_tumor == 0:
        return {
            "mean_confidence": 1.0,
            "uncertain_voxel_count": 0,
            "uncertain_fraction": 0.0,
            "per_class": {},
        }

    mean_unc = unc[tumor_mask].mean().item()
    uncertain_mask = (unc > uncertain_threshold) & tumor_mask
    n_uncertain = uncertain_mask.sum().item()

    per_class = {}
    for c in range(1, NUM_CLASSES):
        class_mask = pred == c
        n = class_mask.sum().item()
        if n > 0:
            avg_unc = unc[class_mask].mean().item()
            per_class[CLASS_NAMES[c]] = {
                "voxels": n,
                "mean_confidence": round(1.0 - avg_unc, 4),
            }

    return {
        "mean_confidence"       : round(1.0 - mean_unc, 4),
        "uncertain_voxel_count" : n_uncertain,
        "uncertain_fraction"    : round(n_uncertain / n_tumor, 4),
        "per_class"             : per_class,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run: python -m src.cranovision.inference.mc_dropout
    from ..config import MODELS_DIR
    from ..data import get_splits
    from .predict import load_model

    print("=" * 60)
    print("CranioVision — mc_dropout.py smoke test")
    print("=" * 60)

    ckpt = MODELS_DIR / "attention_unet_best.pth"
    model = load_model("attention_unet", ckpt)

    _, _, test_cases = get_splits(verbose=False)
    case = test_cases[0]

    # Use only 5 samples for smoke test (speed)
    result = mc_dropout_predict(model, case, n_samples=5, verbose=True)

    print(f"\n─── Result ───")
    print(f"Case ID         : {result['case_id']}")
    print(f"N samples       : {result['n_samples']}")
    print(f"Mean probs shape: {tuple(result['mean_probs'].shape)}")
    print(f"Uncertainty shape: {tuple(result['uncertainty'].shape)}")
    print(f"Uncertainty range: [{result['uncertainty'].min():.4f}, "
          f"{result['uncertainty'].max():.4f}]")

    summary = summarize_confidence(result)
    print(f"\n─── Confidence summary ───")
    print(f"Mean confidence       : {summary['mean_confidence']:.4f}")
    print(f"Uncertain voxels      : {summary['uncertain_voxel_count']:,}")
    print(f"Uncertain fraction    : {summary['uncertain_fraction']*100:.2f}% of tumor")
    print(f"Per-class confidence  :")
    for name, stats in summary['per_class'].items():
        print(f"  {name:20s}: {stats['mean_confidence']:.4f}  "
              f"({stats['voxels']:,} voxels)")

    print("\n✅ mc_dropout.py works.")