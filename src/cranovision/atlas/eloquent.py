"""
CranioVision — Eloquent cortex proximity analysis.

In neurosurgery, "eloquent cortex" means brain regions where damage causes
serious functional deficits — primarily:

  - Primary motor cortex (Precentral Gyrus): movement
  - Primary sensory cortex (Postcentral Gyrus): sensation
  - Broca's area (Inferior Frontal Gyrus, pars opercularis): speech production
  - Wernicke's area (Superior Temporal Gyrus posterior): speech comprehension
  - Primary visual cortex (Intracalcarine, Occipital Pole): vision

When planning tumor surgery, the surgeon NEEDS to know how close the tumor
is to these regions. A tumor 30mm from motor cortex is operable. A tumor
2mm from motor cortex requires awake-craniotomy with intra-op stimulation
mapping. This is one of the most clinically actionable outputs CranioVision
can provide.

How we compute it
-----------------
For each eloquent region:
  1. Build a binary mask of that region in MNI152 space (from Harvard-Oxford)
  2. Compute the Euclidean distance transform — for each voxel in the brain,
     the distance to the nearest voxel of that region
  3. Find the minimum distance value over the tumor's voxels
  4. That minimum is the "shortest distance from tumor edge to eloquent cortex"

Distances are reported in millimeters (because the atlas is 1mm isotropic,
voxel distance == mm distance).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import ants
import numpy as np
from scipy.ndimage import distance_transform_edt

from .download import (
    get_atlas_paths,
    HO_CORTICAL_LABELS,
)


# --- Eloquent region definitions --------------------------------------------

# Map clinical name -> list of Harvard-Oxford cortical label IDs
# Some "regions" span multiple HO labels (e.g., motor cortex includes both
# precentral gyrus and the supplementary motor area).
ELOQUENT_REGIONS: Dict[str, Dict] = {
    "Primary Motor Cortex": {
        "ho_labels": [7],   # Precentral Gyrus
        "function": "Voluntary movement",
        "deficit_if_damaged": "Hemiparesis (weakness on opposite body side)",
    },
    "Supplementary Motor Area": {
        "ho_labels": [26],  # Juxtapositional Lobule Cortex
        "function": "Movement planning",
        "deficit_if_damaged": "Akinetic mutism, complex movement deficits",
    },
    "Primary Somatosensory Cortex": {
        "ho_labels": [17],  # Postcentral Gyrus
        "function": "Body sensation",
        "deficit_if_damaged": "Loss of sensation on opposite body side",
    },
    "Broca's Area (speech production)": {
        "ho_labels": [5, 6],  # IFG pars triangularis + opercularis
        "function": "Speech production (left-dominant in most)",
        "deficit_if_damaged": "Expressive (Broca's) aphasia",
    },
    "Wernicke's Area (speech comprehension)": {
        # STG posterior + Supramarginal posterior + Angular gyrus (left-dominant)
        "ho_labels": [10, 20, 21],
        "function": "Language comprehension",
        "deficit_if_damaged": "Receptive (Wernicke's) aphasia",
    },
    "Primary Visual Cortex": {
        "ho_labels": [24, 48],  # Intracalcarine + Occipital Pole
        "function": "Vision",
        "deficit_if_damaged": "Visual field deficits (homonymous hemianopia)",
    },
}


# --- Distance transform helpers ---------------------------------------------

def _build_region_mask(atlas_arr: np.ndarray, label_ids: list) -> np.ndarray:
    """Build a binary mask containing all voxels with any of the given labels."""
    mask = np.zeros_like(atlas_arr, dtype=bool)
    for lid in label_ids:
        mask |= (atlas_arr == lid)
    return mask


def _min_distance_in_tumor(distance_map: np.ndarray,
                           tumor_mask: np.ndarray) -> float:
    """
    Find the minimum value of distance_map within the tumor region.

    Returns float('inf') if either array is empty in the tumor region.
    """
    if not tumor_mask.any():
        return float("inf")
    return float(distance_map[tumor_mask].min())


# --- Main API ---------------------------------------------------------------

def compute_eloquent_distance(
    warped_mask: Union[ants.ANTsImage, str, Path],
    verbose: bool = False,
) -> Dict[str, Dict]:
    """
    For each eloquent brain region, compute the minimum distance from tumor
    edge to that region.

    Parameters
    ----------
    warped_mask : tumor mask in MNI152 space (output of register_patient)
    verbose     : print results

    Returns
    -------
    dict mapping region name -> {
        "distance_mm": float,
        "function": str,
        "deficit_if_damaged": str,
        "risk_level": "high" | "moderate" | "low" | "minimal",
        "involved": bool   (True if tumor overlaps the region)
    }
    """
    # Load mask
    if isinstance(warped_mask, (str, Path)):
        mask_img = ants.image_read(str(warped_mask))
    else:
        mask_img = warped_mask

    tumor_arr = (mask_img.numpy() > 0)
    if not tumor_arr.any():
        return {
            name: {
                "distance_mm": float("inf"),
                "function": info["function"],
                "deficit_if_damaged": info["deficit_if_damaged"],
                "risk_level": "n/a",
                "involved": False,
            }
            for name, info in ELOQUENT_REGIONS.items()
        }

    # Voxel spacing (assume isotropic 1mm for MNI atlas — sanity check)
    spacing = np.array(mask_img.spacing)
    if not np.allclose(spacing, [1, 1, 1], atol=0.01):
        # If non-isotropic, distance_transform_edt accepts a `sampling` arg
        sampling = tuple(spacing)
    else:
        sampling = (1.0, 1.0, 1.0)

    # Load cortical atlas
    atlas_paths = get_atlas_paths()
    cort = ants.image_read(str(atlas_paths["harvard_oxford_cortical"]))
    cort_arr = cort.numpy().astype(np.int32)

    # Sanity check
    if cort_arr.shape != tumor_arr.shape:
        raise ValueError(
            f"Atlas / mask shape mismatch: {cort_arr.shape} vs {tumor_arr.shape}. "
            f"Mask must be in MNI152 1mm space."
        )

    results: Dict[str, Dict] = {}

    for region_name, info in ELOQUENT_REGIONS.items():
        # Build binary mask for this region
        region_mask = _build_region_mask(cort_arr, info["ho_labels"])

        if not region_mask.any():
            # Region not present in atlas (shouldn't happen, but be safe)
            results[region_name] = {
                "distance_mm": float("inf"),
                "function": info["function"],
                "deficit_if_damaged": info["deficit_if_damaged"],
                "risk_level": "unknown",
                "involved": False,
            }
            continue

        # Distance transform: distance from each voxel to nearest region voxel.
        # We use the *complement* — distance_transform_edt computes distance
        # to the nearest 0 in its input. So we pass ~region_mask, which is 0
        # at the region and 1 elsewhere. The result tells us "how far is each
        # voxel from the eloquent region."
        dist_map = distance_transform_edt(~region_mask, sampling=sampling)

        # Minimum distance within the tumor
        d_min = _min_distance_in_tumor(dist_map, tumor_arr)

        # Tumor-region overlap
        overlap = bool((tumor_arr & region_mask).any())

        # Risk classification
        # Standard neurosurgical safety margins:
        #   - <5 mm: high risk (intra-op mapping required)
        #   - 5-10 mm: moderate risk (careful planning)
        #   - 10-20 mm: low risk
        #   - >20 mm: minimal risk
        if overlap or d_min < 5:
            risk = "high"
        elif d_min < 10:
            risk = "moderate"
        elif d_min < 20:
            risk = "low"
        else:
            risk = "minimal"

        results[region_name] = {
            "distance_mm": round(d_min, 2),
            "function": info["function"],
            "deficit_if_damaged": info["deficit_if_damaged"],
            "risk_level": risk,
            "involved": overlap,
        }

    if verbose:
        print(get_eloquent_summary(results))

    return results


# --- Pretty printing --------------------------------------------------------

def get_eloquent_summary(results: Dict) -> str:
    """Render eloquent-distance results as a clinical-report-style string."""
    lines = []
    lines.append("=" * 72)
    lines.append("ELOQUENT CORTEX PROXIMITY ANALYSIS")
    lines.append("=" * 72)
    lines.append(f"{'Region':<42}{'Distance (mm)':>16}{'Risk':>14}")
    lines.append("-" * 72)

    # Sort by distance ascending so closest regions appear first
    sorted_regions = sorted(results.items(),
                             key=lambda x: x[1]["distance_mm"])

    for name, info in sorted_regions:
        d = info["distance_mm"]
        d_str = "involved" if info["involved"] else f"{d:.1f}"
        risk_marker = {
            "high": "[!] HIGH",
            "moderate": "[*] MOD",
            "low": " - low",
            "minimal": "   minimal",
            "n/a": " n/a",
            "unknown": " ?",
        }.get(info["risk_level"], info["risk_level"])

        lines.append(f"{name:<42}{d_str:>16}{risk_marker:>14}")

    lines.append("-" * 72)
    lines.append("Risk thresholds: HIGH <5mm,  MODERATE 5-10mm,  LOW 10-20mm,  MINIMAL >20mm")
    lines.append("=" * 72)

    # If anything is high risk, add a clinical note
    high_risk = [name for name, info in results.items()
                 if info["risk_level"] == "high"]
    if high_risk:
        lines.append("")
        lines.append("CLINICAL NOTE — Tumor is in or near eloquent cortex:")
        for name in high_risk:
            info = results[name]
            lines.append(f"  - {name}")
            lines.append(f"    Function: {info['function']}")
            lines.append(f"    Risk if damaged: {info['deficit_if_damaged']}")
        lines.append("")
        lines.append("Recommend: pre-op functional MRI / awake craniotomy planning.")
        lines.append("=" * 72)

    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    """Smoke test on a cached registration."""
    from ..data import get_splits
    from .registration import load_cached_registration, register_patient

    _, _, test_cases = get_splits(verbose=False)
    case = test_cases[0]
    case_id = case["case_id"]

    cached = load_cached_registration(case_id)
    if cached is None:
        print(f"No cached registration for {case_id}. Running...")
        cached = register_patient(
            case_id=case_id,
            t1_path=case["t1n"],
            tumor_mask_path=case["label"],
            verbose=True,
        )

    print()
    eloquent = compute_eloquent_distance(
        warped_mask=cached["warped_mask"],
        verbose=True,
    )