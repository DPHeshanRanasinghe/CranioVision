"""
CranioVision — Grad-CAM cross-architecture reliability study.

Reproducibly (re)computes the informal claim asserted in pipeline.py's module
docstring, README.md, and clinical_report.py ("Attention U-Net produces
9-15x signal-to-background Grad-CAM heatmaps; SwinUNETR/nnU-Net are
unreliable"), using metrics with a defined methodology instead of visual
inspection. See research/PROJECT_NOTES.md for why this script exists: that
claim currently has no computation script anywhere in the tracked repo.

Metrics (per case, per model, per tumor class)
-----------------------------------------------
signal_background_ratio
    Mean heatmap activation inside the predicted tumor region divided by
    mean activation in the rest of the brain. Reproduces the original
    informal "9-15x"-style number, now with a defined formula.

concentration_ratio
    Fraction of total heatmap energy (sum of activations) contained inside
    the predicted tumor region. Captures "diffuse vs. concentrated" — a
    heatmap can have a high signal_background_ratio while still spreading
    most of its mass outside the tumor if the background is very large.

deletion_aopc
    Perturbation-based faithfulness (Area Over the Perturbation Curve,
    deletion variant). Ranks the 128^3 Grad-CAM patch's voxels by heatmap
    value, progressively zeroes the top-k% (k in DELETION_FRACTIONS), and
    re-runs a direct model forward pass at each step. AOPC is the mean drop
    in target-class confidence across steps. Unlike the two metrics above,
    this is causal: it asks whether the voxels the heatmap ranks as
    important are ones the model actually depends on, not just spatially
    correlated with the tumor. This is the metric that should anchor the
    paper's claims — see research/related_work.md's "terminology traps" note
    on why signal_background_ratio alone is not a faithfulness metric.

This CANNOT be run in a bare checkout of this repo. It requires trained
checkpoints in MODELS_DIR (attention_unet_best.pth, swin_unetr_best.pth,
nnunet_best.pth) and the BraTS test split under RAW_DATA_DIR — both are
gitignored and were not present when this script was written. Run it wherever
those exist (local GPU or Kaggle), then hand the output JSON back for the
paper's Results section.

Usage
-----
    python -m src.cranovision.analysis.gradcam_reliability
    python -m src.cranovision.analysis.gradcam_reliability --n-cases 10
    python -m src.cranovision.analysis.gradcam_reliability --models attention_unet,nnunet
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR, USE_AMP
from ..data import get_splits
from ..inference.grad_cam import compute_grad_cam
from ..inference.predict import load_model
from ..pipeline import PIPELINE_MODELS

TARGET_CLASSES = (1, 2, 3)  # edema, enhancing, necrotic (see config.CLASS_NAMES)
DELETION_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 1 — signal-to-background ratio
# ══════════════════════════════════════════════════════════════════════════════

def signal_background_ratio(
    heatmap: np.ndarray, tumor_mask: np.ndarray, brain_mask: np.ndarray,
) -> float:
    """
    Mean heatmap value inside the predicted tumor region divided by mean
    heatmap value in the rest of the brain (background = brain minus tumor).

    Returns NaN when there's no predicted tumor or no background brain tissue
    to compare against, so callers can drop those cases instead of silently
    biasing the aggregate.
    """
    tumor = tumor_mask.astype(bool)
    background = brain_mask.astype(bool) & ~tumor

    if not tumor.any() or not background.any():
        return float("nan")

    signal = heatmap[tumor].mean()
    bg = heatmap[background].mean()
    if bg <= 1e-8:
        return float("nan")

    return float(signal / bg)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 2 — concentration ratio
# ══════════════════════════════════════════════════════════════════════════════

def concentration_ratio(heatmap: np.ndarray, tumor_mask: np.ndarray) -> float:
    """
    Fraction of total heatmap energy that falls inside the predicted tumor
    region. 1.0 = perfectly concentrated on the tumor; near 0 = heatmap mass
    is spread across the whole brain regardless of tumor location.
    """
    total = heatmap.sum()
    if total <= 1e-8:
        return float("nan")
    return float(heatmap[tumor_mask.astype(bool)].sum() / total)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 3 — deletion AOPC (perturbation-based faithfulness)
# ══════════════════════════════════════════════════════════════════════════════

def deletion_aopc(
    model: torch.nn.Module,
    image_patch: torch.Tensor,
    heatmap_patch: np.ndarray,
    target_class: int,
    target_mask: torch.Tensor,
    fractions: Sequence[float] = DELETION_FRACTIONS,
) -> float:
    """
    Deletion-variant Area Over the Perturbation Curve.

    image_patch    : (C, D, H, W) preprocessed patch, matches PATCH_SIZE —
                      forwarded directly through the model (no sliding
                      window needed), same convention GradCAM3D.generate uses.
    heatmap_patch   : (D, H, W) Grad-CAM values for this class, patch-local.
    target_mask     : (D, H, W) bool tensor — the region the heatmap was
                       scored against (use_predicted_mask region in
                       compute_grad_cam). Confidence is measured over this
                       region so the metric matches what the heatmap explains.
    """
    device = next(model.parameters()).device

    if target_mask is None or not target_mask.any():
        return float("nan")

    def _confidence(patch: torch.Tensor) -> float:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
                logits = model(patch.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).detach().cpu()
        return float(probs[0, target_class][target_mask].mean().item())

    baseline_conf = _confidence(image_patch)
    if baseline_conf <= 1e-8:
        return float("nan")

    order = np.argsort(heatmap_patch, axis=None)[::-1]  # descending importance
    n_voxels = heatmap_patch.size

    perturbed = image_patch.clone()
    flat = perturbed.view(perturbed.shape[0], -1)  # view shares storage with perturbed

    drops: List[float] = []
    n_done = 0
    for frac in fractions:
        n_target = int(frac * n_voxels)
        idx = order[n_done:n_target]
        flat[:, idx] = 0.0
        n_done = n_target

        conf = _confidence(perturbed)
        drops.append((baseline_conf - conf) / baseline_conf)

    return float(np.mean(drops))


# ══════════════════════════════════════════════════════════════════════════════
# PER-CASE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_case_metrics(
    model_name: str,
    model: torch.nn.Module,
    case: Dict,
    target_classes: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    grad_result = compute_grad_cam(
        model=model, case_dict=case, model_name=model_name,
        target_classes=target_classes, use_predicted_mask=True, verbose=False,
    )

    image = grad_result["image"]          # (C, D, H, W) cpu
    full_pred = grad_result["pred"]        # (D, H, W) cpu
    d0, h0, w0 = grad_result["patch_offset"]
    pd, ph, pw = grad_result["patch_size"]

    image_patch = image[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
    pred_patch = full_pred[d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
    brain_mask_full = (image.abs().sum(dim=0) > 0).numpy()

    metrics: Dict[int, Dict[str, float]] = {}
    for cls in target_classes:
        heatmap_full = grad_result["heatmaps"][cls].numpy()
        heatmap_patch = grad_result["heatmaps_patch"][cls].numpy()
        tumor_mask_full = (full_pred == cls).numpy()
        target_mask_patch = pred_patch == cls

        metrics[cls] = {
            "signal_background_ratio": signal_background_ratio(
                heatmap_full, tumor_mask_full, brain_mask_full,
            ),
            "concentration_ratio": concentration_ratio(heatmap_full, tumor_mask_full),
            "deletion_aopc": deletion_aopc(
                model=model, image_patch=image_patch, heatmap_patch=heatmap_patch,
                target_class=cls, target_mask=target_mask_patch,
            ),
            "n_tumor_voxels": int(tumor_mask_full.sum()),
        }

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STUDY ORCHESTRATION — all models, full (or truncated) test split
# ══════════════════════════════════════════════════════════════════════════════

METRIC_NAMES = ("signal_background_ratio", "concentration_ratio", "deletion_aopc")


def run_reliability_study(
    model_names: Optional[List[str]] = None,
    n_cases: Optional[int] = None,
    target_classes: Sequence[int] = TARGET_CLASSES,
    verbose: bool = True,
) -> Dict:
    if model_names is None:
        model_names = [
            n for n in PIPELINE_MODELS
            if (MODELS_DIR / PIPELINE_MODELS[n]["ckpt"]).exists()
        ]
    if len(model_names) == 0:
        raise RuntimeError(
            "No model checkpoints found in MODELS_DIR. This script needs "
            "trained checkpoints — see the module docstring."
        )

    _, _, test_cases = get_splits(verbose=False)
    if n_cases is not None:
        test_cases = test_cases[:n_cases]

    raw: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        name: {cls: {m: [] for m in METRIC_NAMES} for cls in target_classes}
        for name in model_names
    }
    case_ids: List[str] = []

    for name in model_names:
        cfg = PIPELINE_MODELS[name]
        if verbose:
            print(f"\n=== {cfg['display']} ({len(test_cases)} cases) ===")

        model = load_model(name, MODELS_DIR / cfg["ckpt"], **cfg["build_kwargs"])
        model.eval()

        for i, case in enumerate(test_cases):
            case_id = case.get("case_id", f"case_{i}")
            if name == model_names[0]:
                case_ids.append(case_id)
            if verbose:
                print(f"  [{i + 1}/{len(test_cases)}] {case_id}")

            case_metrics = _compute_case_metrics(name, model, case, target_classes)
            for cls, m in case_metrics.items():
                for metric in METRIC_NAMES:
                    raw[name][cls][metric].append(m[metric])

        del model
        _free_gpu()

    summary = _summarize(raw, model_names, target_classes)
    return {
        "raw": raw,
        "summary": summary,
        "model_names": model_names,
        "case_ids": case_ids,
        "n_cases": len(test_cases),
        "target_classes": list(target_classes),
    }


def _summarize(
    raw: Dict, model_names: List[str], target_classes: Sequence[int],
) -> Dict:
    from scipy import stats

    summary: Dict = {"per_model": {}, "comparisons": {}}

    for name in model_names:
        summary["per_model"][name] = {}
        for cls in target_classes:
            cls_name = CLASS_NAMES[cls]
            summary["per_model"][name][cls_name] = {}
            for metric in METRIC_NAMES:
                arr = np.array(raw[name][cls][metric], dtype=float)
                valid = arr[~np.isnan(arr)]
                summary["per_model"][name][cls_name][metric] = {
                    "mean": float(valid.mean()) if valid.size else None,
                    "std": float(valid.std()) if valid.size else None,
                    "n_valid": int(valid.size),
                    "n_total": int(arr.size),
                }

    reference = "attention_unet" if "attention_unet" in model_names else model_names[0]
    for other in [n for n in model_names if n != reference]:
        key = f"{reference}_vs_{other}"
        summary["comparisons"][key] = {}
        for cls in target_classes:
            cls_name = CLASS_NAMES[cls]
            summary["comparisons"][key][cls_name] = {}
            for metric in METRIC_NAMES:
                a = np.array(raw[reference][cls][metric], dtype=float)
                b = np.array(raw[other][cls][metric], dtype=float)
                mask = ~(np.isnan(a) | np.isnan(b))
                a_valid, b_valid = a[mask], b[mask]

                if a_valid.size < 2 or np.allclose(a_valid, b_valid):
                    summary["comparisons"][key][cls_name][metric] = {
                        "n_pairs": int(a_valid.size), "statistic": None, "p_value": None,
                    }
                    continue

                statistic, p_value = stats.wilcoxon(a_valid, b_valid)
                summary["comparisons"][key][cls_name][metric] = {
                    "n_pairs": int(a_valid.size),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }

    return summary


def _print_summary(result: Dict) -> None:
    print("\n" + "=" * 72)
    print("Per-model summary (mean ± std)")
    print("=" * 72)
    for name, per_class in result["summary"]["per_model"].items():
        print(f"\n{name}:")
        for cls_name, metrics in per_class.items():
            print(f"  {cls_name}:")
            for metric, stats_ in metrics.items():
                if stats_["mean"] is None:
                    print(f"    {metric:26s}: no valid cases")
                else:
                    print(
                        f"    {metric:26s}: {stats_['mean']:.4f} ± {stats_['std']:.4f} "
                        f"(n={stats_['n_valid']}/{stats_['n_total']})"
                    )

    print("\n" + "=" * 72)
    print("Paired comparisons (Wilcoxon signed-rank, reference vs. other)")
    print("=" * 72)
    for comparison, per_class in result["summary"]["comparisons"].items():
        print(f"\n{comparison}:")
        for cls_name, metrics in per_class.items():
            print(f"  {cls_name}:")
            for metric, stats_ in metrics.items():
                if stats_["p_value"] is None:
                    print(f"    {metric:26s}: not testable (n={stats_['n_pairs']})")
                else:
                    print(
                        f"    {metric:26s}: p={stats_['p_value']:.4g} "
                        f"(n={stats_['n_pairs']})"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-cases", type=int, default=None,
        help="Limit to the first N test cases (default: full test split).",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names (default: all with checkpoints present).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: OUTPUTS_DIR/gradcam_reliability_study.json).",
    )
    args = parser.parse_args()

    model_names = args.models.split(",") if args.models else None
    output_path = Path(args.output) if args.output else OUTPUTS_DIR / "gradcam_reliability_study.json"

    print("=" * 72)
    print("CranioVision — Grad-CAM cross-architecture reliability study")
    print("=" * 72)

    t0 = time.time()
    result = run_reliability_study(model_names=model_names, n_cases=args.n_cases)
    elapsed = time.time() - t0

    _print_summary(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Done in {elapsed:.1f}s. Full results written to {output_path}")
