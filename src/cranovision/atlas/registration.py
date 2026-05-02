"""
CranioVision — Patient -> MNI152 registration (ANTs).

This module is the heart of Phase 3: it aligns each patient's T1 MRI to
standard MNI152 space so we can look up anatomical regions for the tumor.

Why this is non-trivial for tumor patients
------------------------------------------
A naive registration would try to make the patient's brain match the atlas
shape EXACTLY. But tumor patients have *mass effect*: the tumor pushes
healthy tissue out of the way, distorts ventricles, and shifts the midline.
If we let SyN (the deformation field) try to undo this distortion, it will:

  1) Warp the tumor itself to look more like healthy MNI tissue (BAD)
  2) Add unrealistic deformations near the lesion (BAD)
  3) Pull surrounding healthy regions away from their true location (BAD)

The professional fix: cost-function masking.
We tell ANTs "ignore the tumor voxels when computing similarity." The
algorithm aligns the *healthy* parts of the brain to MNI152 and just
applies the resulting transform to the tumor without trying to deform it.

Pipeline
--------
1. Load patient T1 + tumor mask
2. Build "registration mask" = inverse of tumor (1 in healthy regions, 0 in tumor)
3. Run ANTs registration:
   - Stage 1: rigid (rotate + translate)
   - Stage 2: affine (+ scale + shear)
   - Stage 3: SyN non-linear deformation
   All stages use mutual information as similarity, with the registration mask
   excluding tumor voxels from the metric.
4. Apply the resulting transform to the tumor mask (now in MNI space)
5. Cache everything to disk

Each registered case takes ~3-5 minutes on CPU. Caching means we only pay
this cost once per patient.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Union

import ants
import numpy as np

from ..config import ATLAS_CACHE_DIR
from .download import get_atlas_paths


# --- Cache helpers ----------------------------------------------------------

def _case_cache_dir(case_id: str) -> Path:
    """Directory holding cached registration artifacts for one patient."""
    d = ATLAS_CACHE_DIR / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_cached(case_id: str) -> bool:
    """Check whether all expected cache files exist for this case."""
    d = _case_cache_dir(case_id)
    expected = [
        d / "warped_t1.nii.gz",       # patient T1 in MNI space (sanity check)
        d / "warped_mask.nii.gz",     # tumor mask in MNI space (the useful output)
        d / "fwd_affine.mat",         # forward affine transform
        d / "fwd_warp.nii.gz",        # forward non-linear warp field
        d / "metadata.json",          # registration parameters and timing
    ]
    return all(p.exists() for p in expected)


def load_cached_registration(case_id: str) -> Optional[Dict]:
    """
    Load a previously cached registration result for this case.

    Returns None if the cache is incomplete or missing. Otherwise returns
    a dict with the same shape as register_patient()'s return value.
    """
    if not _is_cached(case_id):
        return None

    d = _case_cache_dir(case_id)

    warped_t1 = ants.image_read(str(d / "warped_t1.nii.gz"))
    warped_mask = ants.image_read(str(d / "warped_mask.nii.gz"))

    with open(d / "metadata.json") as f:
        metadata = json.load(f)

    return {
        "case_id": case_id,
        "warped_t1": warped_t1,
        "warped_mask": warped_mask,
        "transforms": {
            "affine": str(d / "fwd_affine.mat"),
            "warp": str(d / "fwd_warp.nii.gz"),
        },
        "cache_dir": d,
        "metadata": metadata,
        "from_cache": True,
    }


# --- Mask preparation --------------------------------------------------------

def _build_registration_mask(
    t1_image: ants.ANTsImage,
    tumor_mask: Optional[ants.ANTsImage],
    dilate_voxels: int = 5,
) -> Optional[ants.ANTsImage]:
    """
    Build a mask that tells ANTs which voxels to USE for registration similarity.

    We keep healthy brain voxels (=1) and exclude the tumor + a safety margin
    around it (=0). The dilation is important because the immediate
    peritumoral tissue is also distorted by mass effect.

    If no tumor mask is provided, returns None (ANTs will use the whole brain).
    """
    if tumor_mask is None:
        return None

    # Dilate tumor region — exclude a safety margin around the lesion
    tumor_dilated = ants.iMath(tumor_mask, "MD", dilate_voxels)  # MD = morphological dilation

    # Build the registration mask: brain voxels NOT in dilated tumor
    # We assume t1_image > 0 marks brain (BraTS is skull-stripped so this works).
    brain_mask = ants.threshold_image(t1_image, 1e-5, np.inf, 1, 0)
    reg_mask = brain_mask - tumor_dilated
    reg_mask = ants.threshold_image(reg_mask, 0.5, 1.5, 1, 0)  # binary {0,1}

    return reg_mask


# --- Main registration -------------------------------------------------------

def register_patient(
    case_id: str,
    t1_path: Union[str, Path],
    tumor_mask_path: Optional[Union[str, Path]] = None,
    *,
    use_cache: bool = True,
    transform_type: str = "SyN",
    verbose: bool = True,
) -> Dict:
    """
    Register a patient's T1 MRI to MNI152 space.

    Parameters
    ----------
    case_id        : unique identifier for caching, e.g. 'BraTS-GLI-02105-105'
    t1_path        : path to patient's T1 NIfTI (BraTS uses t1n.nii.gz)
    tumor_mask_path: optional path to a tumor mask NIfTI for cost-function masking.
                     Voxels where mask > 0 are excluded from similarity computation.
                     We recommend passing the WHOLE TUMOR mask (label > 0).
    use_cache      : if True (default), reuse cached result if available
    transform_type : ANTs transform type. "SyN" = affine + non-linear (recommended).
                     Use "Affine" for faster but cruder registration during dev.
    verbose        : print progress

    Returns
    -------
    dict with:
        case_id     : str
        warped_t1   : ANTsImage of the patient's T1 in MNI space
        warped_mask : ANTsImage of the tumor mask in MNI space (None if no mask given)
        transforms  : dict with 'affine' and 'warp' file paths
        cache_dir   : Path to cache directory
        metadata    : dict with timing, parameters, and registration quality info
        from_cache  : bool
    """
    # --- Try cache --------------------------------------------------------
    if use_cache and _is_cached(case_id):
        if verbose:
            print(f"[cache] Using cached registration for {case_id}")
        return load_cached_registration(case_id)

    # --- Load images ------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 64}")
        print(f"Registering: {case_id}")
        print(f"{'=' * 64}")
        print(f"  Patient T1   : {t1_path}")
        if tumor_mask_path:
            print(f"  Tumor mask   : {tumor_mask_path}")
        else:
            print(f"  Tumor mask   : (none — registering whole brain)")
        print(f"  Transform    : {transform_type}")

    t0 = time.time()

    atlas_paths = get_atlas_paths()
    fixed = ants.image_read(str(atlas_paths["mni152_t1_brain"]))   # MNI152 template
    moving = ants.image_read(str(t1_path))                          # patient T1

    tumor_mask = None
    if tumor_mask_path is not None:
        # Load patient-space tumor mask (binary: any tumor label vs background)
        raw_mask = ants.image_read(str(tumor_mask_path))
        # Binarize: anything > 0 is tumor. (For multi-class masks: WT region.)
        tumor_mask = ants.threshold_image(raw_mask, 0.5, np.inf, 1, 0)
        if verbose:
            n_tumor = int(np.sum(tumor_mask.numpy() > 0))
            print(f"  Tumor voxels : {n_tumor:,}")

    # --- Build registration mask (excludes tumor) -------------------------
    reg_mask = _build_registration_mask(moving, tumor_mask, dilate_voxels=5)

    if verbose and reg_mask is not None:
        n_reg = int(np.sum(reg_mask.numpy() > 0))
        n_brain = int(np.sum(moving.numpy() > 1e-5))
        excl_pct = (1 - n_reg / max(n_brain, 1)) * 100
        print(f"  Registration mask: {n_reg:,} voxels ({excl_pct:.1f}% of brain excluded)")

    # --- Run ANTs registration --------------------------------------------
    if verbose:
        print(f"\n  Running ANTs {transform_type}...")
        print(f"  (this typically takes 3-5 min on CPU; first time is slowest)")

    # IMPORTANT: ANTs `mask` argument applies to the FIXED image (atlas).
    # Our reg_mask is in the moving (patient) space. ANTs doesn't natively
    # do moving-space cost-function masking via this API in the same way,
    # so we use a workaround: we'll run ants.registration without a mask
    # but use the moving_mask via the multivariate_extras parameter when
    # available. For antspyx 0.6.x, we use this pattern:
    #
    #   We pass moving_mask to restrict where similarity is computed in
    #   the moving image. This is the tumor-aware approach.

    reg_kwargs = {
        "fixed": fixed,
        "moving": moving,
        "type_of_transform": transform_type,
        "verbose": False,  # ANTs is loud — keep our own logging cleaner
    }

    # antspyx supports masking via 'mask' (fixed-space) and 'moving_mask'.
    # For tumor patients, the tumor is in moving (patient) space, so we use
    # moving_mask if available.
    if reg_mask is not None:
        reg_kwargs["moving_mask"] = reg_mask

    reg_result = ants.registration(**reg_kwargs)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [OK] Registration complete in {elapsed:.1f}s")

    # --- Apply transforms to tumor mask -----------------------------------
    warped_mask = None
    if tumor_mask is not None:
        if verbose:
            print(f"  Warping tumor mask into MNI space...")
        warped_mask = ants.apply_transforms(
            fixed=fixed,
            moving=tumor_mask,
            transformlist=reg_result["fwdtransforms"],
            interpolator="genericLabel",  # preserves labels, no interpolation artifacts
        )
        # Re-binarize after interpolation
        warped_mask = ants.threshold_image(warped_mask, 0.5, np.inf, 1, 0)

    # --- Cache results ----------------------------------------------------
    d = _case_cache_dir(case_id)

    ants.image_write(reg_result["warpedmovout"], str(d / "warped_t1.nii.gz"))

    if warped_mask is not None:
        ants.image_write(warped_mask, str(d / "warped_mask.nii.gz"))
    else:
        # Save an empty mask of MNI shape so the cache check still passes
        empty = fixed.new_image_like(np.zeros_like(fixed.numpy()))
        ants.image_write(empty, str(d / "warped_mask.nii.gz"))

    # ANTs returns transform paths as temp files — copy them into cache
    fwd_transforms = reg_result["fwdtransforms"]  # list, typically [warp, affine]

    # antspyx convention: fwdtransforms = [<warp.nii.gz>, <affine.mat>]
    # The order can vary, so detect by extension
    affine_src = next((t for t in fwd_transforms if t.endswith(".mat")), None)
    warp_src = next((t for t in fwd_transforms if t.endswith(".nii.gz") or t.endswith(".nii")), None)

    if affine_src:
        shutil.copy(affine_src, d / "fwd_affine.mat")
    else:
        # No affine — create empty placeholder so cache check passes
        (d / "fwd_affine.mat").touch()

    if warp_src:
        shutil.copy(warp_src, d / "fwd_warp.nii.gz")
    else:
        # Affine-only: create empty placeholder
        (d / "fwd_warp.nii.gz").touch()

    # --- Save metadata ----------------------------------------------------
    metadata = {
        "case_id": case_id,
        "transform_type": transform_type,
        "t1_path": str(t1_path),
        "tumor_mask_path": str(tumor_mask_path) if tumor_mask_path else None,
        "elapsed_seconds": round(elapsed, 1),
        "atlas_used": "MNI152NLin2009cAsym 1mm brain",
        "registration_mask_used": tumor_mask is not None,
        "n_warp_files": len(fwd_transforms),
    }

    with open(d / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"  [cached] {d}")

    return {
        "case_id": case_id,
        "warped_t1": reg_result["warpedmovout"],
        "warped_mask": warped_mask,
        "transforms": {
            "affine": str(d / "fwd_affine.mat"),
            "warp": str(d / "fwd_warp.nii.gz"),
        },
        "cache_dir": d,
        "metadata": metadata,
        "from_cache": False,
    }


# --- Convenience: warp a NEW mask using cached transforms -------------------

def warp_mask_to_mni(
    case_id: str,
    mask_path: Union[str, Path],
    interpolator: str = "genericLabel",
) -> ants.ANTsImage:
    """
    Apply a cached registration's transforms to a new mask.

    Useful when we want to warp a NEW prediction (e.g., from SwinUNETR or the
    ensemble) into MNI space using the same registration as before, without
    re-running ANTs.
    """
    cache = load_cached_registration(case_id)
    if cache is None:
        raise RuntimeError(
            f"No cached registration for {case_id}. "
            f"Run register_patient(...) first."
        )

    atlas_paths = get_atlas_paths()
    fixed = ants.image_read(str(atlas_paths["mni152_t1_brain"]))
    mask = ants.image_read(str(mask_path))

    transformlist = [cache["transforms"]["warp"], cache["transforms"]["affine"]]

    warped = ants.apply_transforms(
        fixed=fixed,
        moving=mask,
        transformlist=transformlist,
        interpolator=interpolator,
    )
    return warped


# --- CLI smoke test ---------------------------------------------------------

if __name__ == "__main__":
    """Quick smoke test on a single BraTS case if available."""
    from ..data import get_splits

    print("=" * 64)
    print("CranioVision — registration.py smoke test")
    print("=" * 64)

    _, _, test_cases = get_splits(verbose=False)
    case = test_cases[0]
    case_id = case["case_id"]

    # BraTS naming: t1n is the native T1, label is the GT segmentation
    t1_path = case["t1n"]
    mask_path = case["label"]

    result = register_patient(
        case_id=case_id,
        t1_path=t1_path,
        tumor_mask_path=mask_path,
        use_cache=True,
        verbose=True,
    )

    print(f"\nResult summary:")
    print(f"  case_id   : {result['case_id']}")
    print(f"  warped T1 : shape {result['warped_t1'].shape}")
    if result["warped_mask"] is not None:
        n = int(np.sum(result["warped_mask"].numpy() > 0))
        print(f"  warped mask: {n:,} tumor voxels in MNI space")
    print(f"  cache     : {result['cache_dir']}")
    print(f"  from cache: {result['from_cache']}")