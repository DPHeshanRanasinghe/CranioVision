"""
CranioVision — Unified pipeline orchestrator.

Single entry point that runs the entire CranioVision analysis on one MRI
case, with progress callbacks suitable for streaming to a web frontend.

Designed for deployment-first thinking:
- One model on GPU/CPU at a time (memory-efficient)
- Progress callbacks for SSE streaming
- Lazy XAI: anatomy + 4 predictions upfront, Grad-CAM only on demand
- Returns clean dicts that JSON-serialize for HTTP response

XAI architecture
----------------
The XAI explainer is decoupled from the prediction models. We always use
Attention U-Net as the explainer regardless of which model produced the
prediction the user is looking at. This is a deliberate architectural
choice — empirical analysis showed Attention U-Net produces strong
Grad-CAM heatmaps (9-15x signal-to-background) for all three tumor classes,
while SwinUNETR and nnU-Net produce inconsistent or broken heatmaps.

Atlas architecture
------------------
ANTs registration of patient T1 -> MNI152 runs ONCE (cached for demo cases).
After registration, each model's prediction is warped from preprocessed
patient space into MNI using the same cached forward transforms (with
nearestNeighbor interpolation, since predictions are integer label volumes).
Anatomical context (Harvard-Oxford lobes/regions, hemisphere lateralization)
and eloquent-cortex distances are then computed INDEPENDENTLY per prediction.
This honours the product premise that AI provides the segmentation — atlas
results genuinely reflect what each model predicted, not the ground truth.

Usage
-----
    from src.cranovision.pipeline import run_full_analysis, compute_xai_for_model

    def progress(stage, percent, message):
        print(f"[{percent:3d}%] {stage}: {message}")

    result = run_full_analysis(case_dict=case, progress_fn=progress)
    xai = compute_xai_for_model(model_name="swin_unetr", case_dict=case)
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import (
    DEVICE, USE_AMP, MODELS_DIR, OUTPUTS_DIR, CLASS_NAMES,
)
from .data import get_val_transforms
from .inference import (
    load_model,
    weights_from_val_dice,
    compute_region_volumes,
    make_inferer,
    compute_grad_cam,
)
from .training.metrics import compute_case_dice, compute_brats_region_dice


# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

ProgressFn = Callable[[str, int, str], None]


def _noop_progress(stage: str, pct: int, msg: str) -> None:
    pass


# -----------------------------------------------------------------------------
# MODEL REGISTRY
# -----------------------------------------------------------------------------

PIPELINE_MODELS = {
    "attention_unet": {
        "ckpt": "attention_unet_best.pth",
        "build_kwargs": {},
        "display": "Attention U-Net",
    },
    "swin_unetr": {
        "ckpt": "swin_unetr_best.pth",
        "build_kwargs": {"feature_size": 48, "use_checkpoint": False},
        "display": "SwinUNETR",
    },
    "nnunet": {
        "ckpt": "nnunet_best.pth",
        "build_kwargs": {
            "filters": (32, 64, 128, 256, 320, 320),
            "deep_supervision": False,
        },
        "display": "nnU-Net DynUNet",
    },
}

XAI_EXPLAINER_NAME = "attention_unet"


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_val_dice(model_names: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in model_names:
        hp = OUTPUTS_DIR / f"{name}_history.json"
        if hp.exists():
            with open(hp) as f:
                hist = json.load(f)
            out[name] = float(hist.get("best_dice", 1.0))
        else:
            out[name] = 1.0
    return out


# -----------------------------------------------------------------------------
# ATLAS HELPER — per-model warping + analysis
# -----------------------------------------------------------------------------

def _run_atlas_analysis(
    case_dict: Dict,
    case_id: str,
    progress: ProgressFn,
    prediction_names: List[str],
    predictions: Optional[Dict[str, "torch.Tensor"]] = None,
    preprocessed_affine: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Run atlas registration once, then warp each per-model prediction into MNI
    using the cached forward transforms and run anatomy + eloquent INDEPENDENTLY
    on each warped prediction.

    Falls back to the legacy shared-GT analysis when ``predictions`` or
    ``preprocessed_affine`` is not provided (e.g. older callers, or cases where
    the affine could not be recovered from the MetaTensor).
    """
    atlas_results: Dict[str, Dict] = {}

    try:
        from .atlas import (
            register_patient,
            analyze_tumor_anatomy,
            compute_eloquent_distance,
        )

        # Resolve T1n path
        if isinstance(case_dict.get("image"), (list, tuple)):
            t1n_path = case_dict["image"][0]
        else:
            t1n_path = case_dict.get("t1n")

        if t1n_path is None:
            raise RuntimeError("Cannot find T1n path in case_dict")

        # GT mask used only for the patient->MNI alignment when available
        # (it gives ANTs an extra anatomical anchor). Anatomy/eloquent are
        # NOT computed from this mask — they're computed from each model's
        # warped prediction below.
        registration_mask_path = case_dict.get("label")

        progress("atlas", 80, "Registering patient T1 to MNI152")

        reg = register_patient(
            case_id=case_id,
            t1_path=t1n_path,
            tumor_mask_path=registration_mask_path,
            use_cache=True,
            verbose=False,
        )

        per_model_mode = (
            predictions is not None
            and preprocessed_affine is not None
            and reg.get("transforms") is not None
        )

        registration_source = (
            "ground_truth_mask" if registration_mask_path else "no_mask"
        )

        if per_model_mode:
            import ants
            import nibabel as nib
            import tempfile
            from .atlas.download import get_atlas_paths

            transforms = reg["transforms"]
            transform_list = [transforms["warp"], transforms["affine"]]
            atlas_paths = get_atlas_paths()
            mni_fixed = ants.image_read(str(atlas_paths["mni152_t1_brain"]))

            affine_np = np.asarray(preprocessed_affine, dtype=np.float64)
            if affine_np.ndim == 3:
                affine_np = affine_np[0]

            n_preds = max(1, len(predictions))
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp_dir = Path(tmp_str)
                for i, (name, pred_tensor) in enumerate(predictions.items()):
                    pct = 82 + int((i / n_preds) * 14)
                    progress(
                        "atlas", pct,
                        f"Warping {name} to MNI + anatomical analysis",
                    )

                    pred_np = pred_tensor.detach().cpu().numpy().astype(np.int16)
                    tmp_pred = tmp_dir / f"{name}_preprocessed.nii.gz"
                    nib.save(
                        nib.Nifti1Image(pred_np, affine_np),
                        str(tmp_pred),
                    )

                    moving = ants.image_read(str(tmp_pred))
                    warped = ants.apply_transforms(
                        fixed=mni_fixed,
                        moving=moving,
                        transformlist=transform_list,
                        interpolator="nearestNeighbor",
                    )

                    anatomy = analyze_tumor_anatomy(
                        warped_mask=warped,
                        verbose=False,
                    )
                    eloquent = compute_eloquent_distance(
                        warped_mask=warped,
                        verbose=False,
                    )

                    atlas_results[name] = {
                        "anatomy": anatomy,
                        "eloquent": _serialize_eloquent(eloquent),
                        "registration_source": registration_source,
                        "shared_across_predictions": False,
                    }
        else:
            # Legacy path — run once on the cached GT-warped mask and attach
            # the same result to every prediction. Used only when callers
            # don't supply per-model predictions / affine.
            progress("atlas", 92, "Computing anatomical context")

            anatomy = analyze_tumor_anatomy(
                warped_mask=reg["warped_mask"],
                verbose=False,
            )
            eloquent = compute_eloquent_distance(
                warped_mask=reg["warped_mask"],
                verbose=False,
            )

            shared_result = {
                "anatomy": anatomy,
                "eloquent": _serialize_eloquent(eloquent),
                "registration_source": registration_source,
                "shared_across_predictions": True,
            }
            for name in prediction_names:
                atlas_results[name] = shared_result

    except Exception as e:
        atlas_results["_error"] = f"{type(e).__name__}: {e}"

    return atlas_results


# -----------------------------------------------------------------------------
# STAGE 1: FULL ANALYSIS — runs upfront when user uploads MRI
# -----------------------------------------------------------------------------

def run_full_analysis(
    case_dict: Dict,
    progress_fn: Optional[ProgressFn] = None,
    include_atlas: bool = True,
    available_models: Optional[List[str]] = None,
) -> Dict:
    """
    Run the upfront analysis pipeline:
      1. Preprocess MRI
      2. Run all 3 models sequentially (one at a time on GPU)
      3. Compute weighted ensemble
      4. Run atlas registration (if enabled)
      5. Compute anatomy + eloquent (shared across predictions)
      6. Compute model agreement statistics

    Does NOT run Grad-CAM — that's lazy via compute_xai_for_model.
    """
    progress = progress_fn or _noop_progress
    t0 = time.time()
    case_id = case_dict.get("case_id", "unknown")

    if available_models is None:
        available_models = [
            n for n in PIPELINE_MODELS
            if (MODELS_DIR / PIPELINE_MODELS[n]["ckpt"]).exists()
        ]
    if len(available_models) == 0:
        raise RuntimeError("No model checkpoints found in MODELS_DIR")

    progress("init", 0, f"Starting analysis for {case_id}")

    # 1) Preprocess
    progress("preprocess", 5, "Loading and normalizing MRI volumes")
    transforms = get_val_transforms()
    sample = transforms(case_dict)
    full_image = sample["image"]
    gt = sample.get("label")
    gt_3d = gt.squeeze(0).cpu() if gt is not None else None

    inferer = make_inferer()

    # 2) Run each model — one at a time on GPU
    per_model_probs: Dict[str, torch.Tensor] = {}
    per_model_preds: Dict[str, torch.Tensor] = {}
    n_models = len(available_models)

    for i, name in enumerate(available_models):
        cfg = PIPELINE_MODELS[name]
        pct = 10 + (i * 50 // n_models)
        progress("inference", pct, f"Running {cfg['display']}...")

        model = load_model(name, MODELS_DIR / cfg["ckpt"], **cfg["build_kwargs"])
        model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                image_gpu = full_image.unsqueeze(0).to(DEVICE)
                logits = inferer(image_gpu, model)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().float()
            pred = probs.argmax(dim=0)

        per_model_probs[name] = probs
        per_model_preds[name] = pred

        del model, logits, image_gpu
        _free_gpu()

    # 3) Compute weighted ensemble
    progress("ensemble", 65, "Computing weighted ensemble")
    val_dice = _load_val_dice(available_models)
    weights = weights_from_val_dice(val_dice)

    weighted_sum = None
    for name, probs in per_model_probs.items():
        w = weights[name]
        if weighted_sum is None:
            weighted_sum = w * probs
        else:
            weighted_sum += w * probs
    ensemble_pred = weighted_sum.argmax(dim=0)
    per_model_preds["ensemble"] = ensemble_pred

    # 4) Per-model metrics
    progress("metrics", 70, "Computing volumes and metrics")
    per_model_metrics: Dict[str, Dict] = {}
    for name, pred in per_model_preds.items():
        m: Dict = {"volumes_cm3": compute_region_volumes(pred)}
        if gt_3d is not None:
            dice = compute_case_dice(pred, gt_3d)
            brats = compute_brats_region_dice(pred, gt_3d)
            m["dice_per_class"] = {
                cn: float(d) for cn, d in zip(CLASS_NAMES[1:], dice)
            }
            m["mean_dice"] = float(np.mean(dice))
            m["brats_regions"] = {k: float(v) for k, v in brats.items()}
        per_model_metrics[name] = m

    # 5) Agreement (only over the individual models)
    #
    # Whole-volume agreement is dominated by background voxels, where every
    # model trivially predicts 0. The clinically interesting number is how
    # often the models AGREE inside the tumor envelope. We report both:
    #
    #   unanimous_fraction       : (all models agree) / (all voxels)
    #   tumor_region_agreement   : (all models agree AND it's a tumor voxel)
    #                              / (any model called it tumor)
    #
    # The denominator for tumor agreement is the union of all model tumor
    # masks, so a region only one model picked up still counts as
    # "disagreement" — that's the desired conservative reading.
    progress("agreement", 75, "Computing model agreement")
    individual_preds = [per_model_preds[n] for n in available_models]
    stacked = torch.stack(individual_preds)
    mode_pred = torch.mode(stacked, dim=0).values
    all_agree = (stacked == mode_pred.unsqueeze(0)).all(dim=0)  # bool
    unanimous = all_agree.float().mean().item()

    any_tumor = (stacked > 0).any(dim=0)             # union of tumor masks
    unanimous_tumor = all_agree & any_tumor          # agreed AND tumor
    n_any_tumor = int(any_tumor.sum().item())
    if n_any_tumor > 0:
        tumor_region_agreement = float(unanimous_tumor.sum().item()) / n_any_tumor
    else:
        tumor_region_agreement = 0.0

    agreement = {
        "unanimous_fraction": round(unanimous, 4),
        "tumor_region_agreement": round(tumor_region_agreement, 4),
        "n_models_compared": len(individual_preds),
    }

    # Affine of the preprocessed grid in patient world coordinates. Needed
    # both by the atlas analysis (to warp each prediction to MNI for anatomy
    # + eloquent) and by the backend mesh extractor (for the 3D viewer).
    # MONAI's MetaTensor tracks the affine through Orientation/Crop/Pad.
    preprocessed_affine = None
    try:
        aff = getattr(full_image, "affine", None)
        if aff is not None:
            preprocessed_affine = (
                aff.detach().cpu().numpy() if hasattr(aff, "detach") else np.asarray(aff)
            )
    except Exception:
        preprocessed_affine = None

    # 6) Atlas analysis — per-model: warp each prediction to MNI, then run
    # anatomy + eloquent independently for each one.
    atlas_results: Dict[str, Dict] = {}
    if include_atlas:
        atlas_results = _run_atlas_analysis(
            case_dict=case_dict,
            case_id=case_id,
            progress=progress,
            prediction_names=list(per_model_preds.keys()),
            predictions=per_model_preds,
            preprocessed_affine=preprocessed_affine,
        )

    progress("done", 100, "Analysis complete")
    elapsed = time.time() - t0

    return {
        "case_id": case_id,
        "predictions": per_model_preds,
        "per_model_metrics": per_model_metrics,
        "agreement": agreement,
        "atlas": atlas_results,
        "weights_used": weights,
        "available_models": available_models,
        "elapsed_seconds": round(elapsed, 1),
        "preprocessed_affine": preprocessed_affine,
    }


def _serialize_eloquent(eloquent: Dict) -> Dict:
    """Make eloquent dict JSON-friendly (handle inf and tensors)."""
    out = {}
    for name, info in eloquent.items():
        d = info.get("distance_mm")
        out[name] = {
            "distance_mm": None if d is None or d == float("inf") else float(d),
            "risk_level": info.get("risk_level"),
            "involved": bool(info.get("involved")),
            "function": info.get("function"),
            "deficit_if_damaged": info.get("deficit_if_damaged"),
        }
    return out


# -----------------------------------------------------------------------------
# STAGE 2: LAZY XAI — runs when user clicks a model's "Explain" button
# -----------------------------------------------------------------------------

def compute_xai_for_model(
    model_name: str,
    case_dict: Dict,
    target_classes: Sequence[int] = (1, 2, 3),
    progress_fn: Optional[ProgressFn] = None,
) -> Dict:
    """
    Compute Grad-CAM heatmaps that explain MRI features driving tumor
    predictions. The model_name argument identifies which PREDICTION the
    user is viewing — but the EXPLAINER used internally is always
    Attention U-Net (validated empirically as the only architecture among
    our three that produces reliable Grad-CAM heatmaps).
    """
    progress = progress_fn or _noop_progress

    if model_name not in (*PIPELINE_MODELS.keys(), "ensemble"):
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(PIPELINE_MODELS.keys()) + ['ensemble']}"
        )

    explainer_cfg = PIPELINE_MODELS[XAI_EXPLAINER_NAME]
    progress("xai_init", 5, f"Loading explainer ({explainer_cfg['display']})")

    explainer = load_model(
        XAI_EXPLAINER_NAME,
        MODELS_DIR / explainer_cfg["ckpt"],
        **explainer_cfg["build_kwargs"],
    )

    progress("xai_compute", 30, "Running patch-based Grad-CAM")
    result = compute_grad_cam(
        model=explainer,
        case_dict=case_dict,
        model_name=XAI_EXPLAINER_NAME,
        target_classes=target_classes,
        use_predicted_mask=True,
        verbose=False,
    )

    del explainer
    _free_gpu()

    result["explainer_model"] = XAI_EXPLAINER_NAME
    result["prediction_being_explained"] = model_name

    progress("xai_done", 100, "Explanation ready")
    return result