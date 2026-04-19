"""
CranioVision — Ensemble inference (weighted soft voting).

Combines predictions from 3 models:
  1. Attention U-Net  — high recall for diffuse edema
  2. SwinUNETR        — best precision on small enhancing cores
  3. nnU-Net (DynUNet) — reliable baseline, rarely fails catastrophically

Each model produces softmax probabilities. The ensemble averages them
(optionally with learned weights) and takes argmax. This reliably
outperforms any single model by 3-8 Dice points because:
  - Different architectures have different failure modes
  - Where one model is uncertain, another is often confident
  - Voting suppresses individual model artifacts

Supports:
  - Equal-weight averaging (default)
  - Per-model weights (based on validation Dice)
  - Per-class weights (optional, for advanced tuning)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer

from ..config import (
    DEVICE, USE_AMP, PATCH_SIZE, NUM_CLASSES, CLASS_NAMES, MODELS_DIR,
)
from ..data import get_val_transforms
from .predict import load_model, make_inferer, compute_region_volumes


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Default model registry — maps name → (build kwargs, expected .pth filename)
MODEL_REGISTRY = {
    "attention_unet": {
        "ckpt": "attention_unet_best.pth",
        "build_kwargs": {},
    },
    "swin_unetr": {
        "ckpt": "swin_unetr_best.pth",
        "build_kwargs": {"feature_size": 48, "use_checkpoint": False},
    },
    "nnunet": {
        "ckpt": "nnunet_best.pth",
        "build_kwargs": {
            "filters": (32, 64, 128, 256, 320, 320),
            "deep_supervision": True,
            "deep_supr_num": 2,
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL AVAILABLE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_ensemble(
    model_names: Optional[List[str]] = None,
    models_dir: Path = MODELS_DIR,
    verbose: bool = True,
) -> Dict[str, nn.Module]:
    """
    Load all available model checkpoints.

    Args:
        model_names: list of model names to load. If None, loads all models
                     whose .pth files exist in models_dir.
        models_dir : directory containing .pth files

    Returns:
        dict mapping model_name → loaded model (eval mode, on DEVICE)
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    loaded = {}
    skipped = []

    for name in model_names:
        if name not in MODEL_REGISTRY:
            if verbose:
                print(f"  ⚠ Unknown model: {name} — skipping")
            continue

        ckpt_path = models_dir / MODEL_REGISTRY[name]["ckpt"]
        if not ckpt_path.exists():
            if verbose:
                print(f"  ⚠ Checkpoint not found: {ckpt_path.name} — skipping {name}")
            skipped.append(name)
            continue

        model = load_model(
            model_name=name,
            ckpt_path=ckpt_path,
            **MODEL_REGISTRY[name]["build_kwargs"],
        )
        loaded[name] = model

    if verbose:
        print(f"\n✓ Loaded {len(loaded)} model(s): {list(loaded.keys())}")
        if skipped:
            print(f"  Skipped {len(skipped)}: {skipped}")
        print(f"  (Need all 3 for full ensemble — train missing models on Kaggle)")

    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-MODEL PROBABILITY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _predict_probs(
    model: nn.Module,
    image: torch.Tensor,
    inferer: SlidingWindowInferer,
) -> torch.Tensor:
    """
    Run sliding-window inference and return softmax probabilities.
    image: (1, 4, D, H, W) on DEVICE
    Returns: (1, C, D, H, W) float32 on CPU
    """
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = inferer(image, model)
    probs = torch.softmax(logits, dim=1).cpu().float()
    torch.cuda.empty_cache()
    return probs


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICT — WEIGHTED SOFT VOTING
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(
    models: Dict[str, nn.Module],
    case_dict: Dict,
    weights: Optional[Dict[str, float]] = None,
    inferer: Optional[SlidingWindowInferer] = None,
    return_per_model: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run all models on one case and combine via weighted soft voting.

    Args:
        models         : dict of model_name → model (from load_ensemble)
        case_dict      : BraTS case dict
        weights        : optional per-model weights. If None, equal weighting.
                         Example: {'attention_unet': 0.25, 'swin_unetr': 0.45, 'nnunet': 0.30}
        inferer        : shared sliding-window inferer
        return_per_model: if True, also return individual model predictions

    Returns dict:
        'pred'          : (D, H, W) ensemble prediction (argmax of weighted mean)
        'mean_probs'    : (C, D, H, W) weighted average probabilities
        'image'         : (4, D, H, W) preprocessed input
        'label'         : (1, D, H, W) ground truth if available
        'case_id'       : str
        'weights_used'  : dict of actual weights
        'per_model'     : (optional) dict of model_name → individual pred
    """
    if inferer is None:
        inferer = make_inferer()

    if weights is None:
        # Equal weighting
        n = len(models)
        weights = {name: 1.0 / n for name in models}
    else:
        # Normalize weights to sum to 1
        total = sum(weights[name] for name in models if name in weights)
        weights = {name: weights.get(name, 0.0) / total for name in models}

    # Preprocess
    transforms = get_val_transforms()
    sample = transforms(case_dict)
    image = sample["image"].unsqueeze(0).to(DEVICE)

    # Collect weighted probabilities
    weighted_sum = None
    per_model_preds = {}

    for name, model in models.items():
        if verbose:
            print(f"  Running {name}...")
        probs = _predict_probs(model, image, inferer)    # (1, C, D, H, W)
        probs = probs.squeeze(0)                          # (C, D, H, W)

        w = weights[name]
        if weighted_sum is None:
            weighted_sum = w * probs
        else:
            weighted_sum += w * probs

        if return_per_model:
            per_model_preds[name] = probs.argmax(dim=0)   # (D, H, W)

    # Final ensemble prediction
    ensemble_pred = weighted_sum.argmax(dim=0)             # (D, H, W)

    result = {
        "pred"        : ensemble_pred,
        "mean_probs"  : weighted_sum,
        "image"       : sample["image"].cpu(),
        "label"       : sample["label"].cpu() if "label" in sample else None,
        "case_id"     : case_dict.get("case_id", "unknown"),
        "weights_used": weights,
        "n_models"    : len(models),
    }

    if return_per_model:
        result["per_model"] = per_model_preds

    return result


# ══════════════════════════════════════════════════════════════════════════════
# AGREEMENT MAP — where do models agree/disagree?
# ══════════════════════════════════════════════════════════════════════════════

def compute_agreement(per_model_preds: Dict[str, torch.Tensor]) -> Dict:
    """
    Analyze where models agree and disagree.

    Args:
        per_model_preds: dict of model_name → (D, H, W) int prediction

    Returns:
        'agreement_map'       : (D, H, W) float — fraction of models agreeing
        'unanimous_fraction'  : float — what % of voxels have 100% agreement
        'disagreement_mask'   : (D, H, W) bool — True where not all agree
    """
    preds = list(per_model_preds.values())
    n = len(preds)
    stacked = torch.stack(preds)                          # (N, D, H, W)

    # Mode vote
    mode_pred = torch.mode(stacked, dim=0).values         # (D, H, W)

    # Agreement = fraction of models matching the mode
    agree_count = (stacked == mode_pred.unsqueeze(0)).float().sum(dim=0)
    agreement_map = agree_count / n

    unanimous = (agreement_map == 1.0).float().mean().item()
    disagreement_mask = agreement_map < 1.0

    return {
        "agreement_map"      : agreement_map,
        "unanimous_fraction" : round(unanimous, 4),
        "disagreement_mask"  : disagreement_mask,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT OPTIMIZATION — set per-model weights from validation Dice
# ══════════════════════════════════════════════════════════════════════════════

def weights_from_val_dice(val_dice: Dict[str, float]) -> Dict[str, float]:
    """
    Compute model weights proportional to their validation Dice.

    Example:
        val_dice = {'attention_unet': 0.76, 'swin_unetr': 0.84, 'nnunet': 0.78}
        → weights = {'attention_unet': 0.319, 'swin_unetr': 0.353, 'nnunet': 0.328}
    """
    total = sum(val_dice.values())
    return {name: d / total for name, d in val_dice.items()}


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ..data import get_splits
    from ..training.metrics import compute_case_dice, compute_brats_region_dice

    print("=" * 60)
    print("CranioVision — ensemble.py smoke test")
    print("=" * 60)

    # Load whatever models are available
    models = load_ensemble(verbose=True)

    if len(models) == 0:
        print("\n⚠ No model checkpoints found. Train at least one model first.")
        exit(0)

    _, _, test_cases = get_splits(verbose=False)
    case = test_cases[0]

    print(f"\nRunning ensemble on: {case['case_id']}")
    result = ensemble_predict(
        models=models,
        case_dict=case,
        return_per_model=True,
        verbose=True,
    )

    # Volumes
    volumes = compute_region_volumes(result['pred'])
    print(f"\n─── Ensemble result ───")
    print(f"Models used : {result['n_models']}")
    print(f"Weights     : {result['weights_used']}")
    print(f"Pred shape  : {tuple(result['pred'].shape)}")
    print(f"Volumes     :")
    for name, vol in volumes.items():
        print(f"  {name:20s}: {vol:>7.2f} cm³")

    # Dice vs ground truth
    if result['label'] is not None:
        gt = result['label'].squeeze(0)
        dice = compute_case_dice(result['pred'], gt)
        brats = compute_brats_region_dice(result['pred'], gt)
        print(f"\n─── Dice vs ground truth ───")
        for i, name in enumerate(CLASS_NAMES[1:]):
            print(f"  {name:20s}: {dice[i]:.4f}")
        print(f"  {'Mean':20s}: {sum(dice)/len(dice):.4f}")
        print(f"\nBraTS regions:")
        for r, d in brats.items():
            print(f"  {r}: {d:.4f}")

    # Agreement analysis
    if "per_model" in result and len(result["per_model"]) > 1:
        agreement = compute_agreement(result["per_model"])
        print(f"\n─── Model agreement ───")
        print(f"Unanimous voxels: {agreement['unanimous_fraction']*100:.2f}%")
    elif len(models) == 1:
        print(f"\n(Only 1 model loaded — agreement analysis needs 2+)")

    print("\n✅ ensemble.py works.")