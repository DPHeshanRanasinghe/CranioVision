"""
CranioVision — Tumor anatomical analysis.

Once a tumor mask has been registered to MNI152 space (see registration.py),
this module looks up which anatomical regions the tumor occupies using the
Harvard-Oxford parcellation.

Output is a structured dict that's easy to render in the clinical report
or send to the frontend.

Example output (simplified):
{
  "primary_region": "Left Temporal Lobe",
  "primary_pct": 67.4,
  "regions_involved": [
      ("Middle Temporal Gyrus, posterior division", 42.1, 52.6),
      ("Inferior Temporal Gyrus, posterior division", 25.3, 31.6),
      ("Superior Temporal Gyrus, posterior division",  9.1, 11.4),
  ],
  "lateralization": "left",
  "left_hemisphere_pct": 95.2,
  "right_hemisphere_pct": 4.8,
  "total_voxels": 2843,
  "total_volume_cm3": 2.84
}
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import ants
import numpy as np

from .download import (
    get_atlas_paths,
    HO_CORTICAL_LABELS,
    HO_SUBCORTICAL_LABELS,
)


# --- Lobe groupings (which regions belong to which lobe) -------------------

# Map cortical region IDs (Harvard-Oxford) -> lobe name
# This is the standard neurology textbook grouping.
LOBE_GROUPS = {
    "Frontal Lobe": [1, 3, 4, 5, 6, 7, 25, 26, 28, 33, 41],
    "Temporal Lobe": [8, 9, 10, 11, 12, 13, 14, 15, 16, 34, 35, 37, 38, 44, 45, 46],
    "Parietal Lobe": [17, 18, 19, 20, 21, 31, 43],
    "Occipital Lobe": [22, 23, 24, 32, 36, 39, 40, 47, 48],
    "Insular Cortex": [2],
    "Cingulate": [29, 30],
    "Operculum": [42],
}

# Subcortical region IDs (offset by +100 to keep them separate from cortical)
SUBCORTICAL_OFFSET = 100


# --- Hemisphere lookup ------------------------------------------------------

def _classify_hemisphere(coords_mm: np.ndarray) -> Tuple[float, float]:
    """
    Classify each voxel as left or right hemisphere based on its mm
    x-coordinate.

    Coordinates here come from `_voxels_to_mni_mm`, which derives them from
    the ANTs image affine. ANTs stores all images in **LPS+** convention
    regardless of the source file orientation, meaning:

        +x --> Left   (patient anatomical left)
        -x --> Right  (patient anatomical right)

    This is the OPPOSITE of the RAS convention used by some other tools
    (NiBabel default, FSL native, etc.). Getting this wrong produces a
    silent L/R flip in the report — verified empirically on BraTS-GLI-02143-102
    where the atlas labelled the dominant region "Right Cerebral White Matter"
    while a RAS-style classifier was reporting "Left 97%".

    x = 0 voxels (midline) are counted as right by convention.

    Parameters
    ----------
    coords_mm : (N, 3) array of voxel coordinates in LPS mm space

    Returns
    -------
    (left_pct, right_pct) : percentages of voxels in each hemisphere
    """
    if len(coords_mm) == 0:
        return 0.0, 0.0
    n_left = int(np.sum(coords_mm[:, 0] > 0))
    n_right = int(np.sum(coords_mm[:, 0] <= 0))
    total = n_left + n_right
    return 100.0 * n_left / total, 100.0 * n_right / total


# --- Voxel -> mm coordinate conversion -------------------------------------

def _voxels_to_mni_mm(image: ants.ANTsImage, voxel_indices: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) voxel indices to MNI millimeter coordinates.

    Uses the image's affine / direction matrix. ANTs stores this internally
    and exposes it via .origin, .spacing, .direction.

    Note: ANTs uses LPS+ orientation by default. MNI templates are typically
    LPS, so x is left/right with origin at AC. We don't need to flip.
    """
    # Build the affine: world = direction * spacing * voxel + origin
    origin = np.array(image.origin)
    spacing = np.array(image.spacing)
    direction = np.array(image.direction).reshape(3, 3)

    # spacing applied as a diagonal scaling
    affine_3x3 = direction * spacing  # (3, 3)
    coords_mm = (voxel_indices @ affine_3x3.T) + origin
    return coords_mm


# --- Region lookup ----------------------------------------------------------

def _build_combined_atlas(cortical: ants.ANTsImage,
                          subcortical: ants.ANTsImage) -> np.ndarray:
    """
    Combine cortical and subcortical atlases into one label map.

    Subcortical labels are offset by SUBCORTICAL_OFFSET to avoid clashing
    with cortical labels. So:
      - Cortical labels live at 1-48
      - Subcortical labels live at 101-121

    Where both atlases assign labels (rare overlap), cortical wins.
    """
    cort = cortical.numpy().astype(np.int32)
    sub = subcortical.numpy().astype(np.int32)

    combined = cort.copy()
    # Where cortical is 0 (background) and subcortical has a label, use subcortical
    sub_offset = np.where(sub > 0, sub + SUBCORTICAL_OFFSET, 0)
    combined = np.where(combined == 0, sub_offset, combined)
    return combined


def _label_name(label_id: int) -> str:
    """Look up a region name by combined label ID."""
    if label_id == 0:
        return "Background / White Matter / CSF"
    if label_id < SUBCORTICAL_OFFSET:
        return HO_CORTICAL_LABELS.get(label_id, f"Unknown cortical ({label_id})")
    sub_id = label_id - SUBCORTICAL_OFFSET
    return HO_SUBCORTICAL_LABELS.get(sub_id, f"Unknown subcortical ({sub_id})")


def _label_to_lobe(label_id: int) -> str:
    """Map a label ID to its lobe (or 'Other'/'Subcortical')."""
    if label_id >= SUBCORTICAL_OFFSET:
        return "Subcortical"
    for lobe, ids in LOBE_GROUPS.items():
        if label_id in ids:
            return lobe
    return "Other Cortical"


# --- Main analysis ----------------------------------------------------------

def analyze_tumor_anatomy(
    warped_mask: Union[ants.ANTsImage, str, Path],
    voxel_volume_mm3: float = 1.0,
    top_n_regions: int = 10,
    verbose: bool = False,
) -> Dict:
    """
    Compute the anatomical distribution of a tumor mask in MNI space.

    Parameters
    ----------
    warped_mask      : tumor mask already registered to MNI152 space
                       (output of register_patient()['warped_mask'])
    voxel_volume_mm3 : volume of one voxel in mm^3 (1.0 for 1mm isotropic MNI)
    top_n_regions    : how many top regions to include in the result
    verbose          : print summary

    Returns
    -------
    dict with anatomical breakdown:
        primary_region   : str — the lobe with most tumor
        primary_pct      : float — % of tumor in that lobe
        regions_involved : list of (region_name, voxel_count, pct_of_tumor)
        lobes            : dict lobe -> (voxel_count, pct_of_tumor)
        lateralization   : "left" | "right" | "bilateral"
        left_hemisphere_pct  : float
        right_hemisphere_pct : float
        total_voxels     : int
        total_volume_cm3 : float
    """
    # Load mask
    if isinstance(warped_mask, (str, Path)):
        mask_img = ants.image_read(str(warped_mask))
    else:
        mask_img = warped_mask

    mask_arr = (mask_img.numpy() > 0)
    total_voxels = int(mask_arr.sum())

    if total_voxels == 0:
        return {
            "primary_region": "No tumor",
            "primary_pct": 0.0,
            "regions_involved": [],
            "lobes": {},
            "lateralization": "n/a",
            "left_hemisphere_pct": 0.0,
            "right_hemisphere_pct": 0.0,
            "total_voxels": 0,
            "total_volume_cm3": 0.0,
        }

    # Load atlases
    atlas_paths = get_atlas_paths()
    cort = ants.image_read(str(atlas_paths["harvard_oxford_cortical"]))
    sub = ants.image_read(str(atlas_paths["harvard_oxford_subcortical"]))

    # Sanity check: atlas and mask must have the same shape
    if cort.shape != mask_img.shape:
        raise ValueError(
            f"Shape mismatch: atlas {cort.shape} vs mask {mask_img.shape}. "
            f"Mask must be in MNI152 space — call register_patient() first."
        )

    # Combined label map
    combined_atlas = _build_combined_atlas(cort, sub)
    tumor_labels = combined_atlas[mask_arr]   # 1D array of labels at tumor voxels

    # Region counts
    unique, counts = np.unique(tumor_labels, return_counts=True)
    region_counts = list(zip(unique.tolist(), counts.tolist()))
    # Sort by count descending
    region_counts.sort(key=lambda x: -x[1])

    # Build region list (skip background = 0)
    regions_involved: List[Tuple[str, int, float]] = []
    for label_id, count in region_counts[:top_n_regions]:
        if label_id == 0:
            continue
        pct = 100.0 * count / total_voxels
        regions_involved.append((_label_name(label_id), int(count), round(pct, 2)))

    # Lobe-level grouping
    lobes: Dict[str, List[int]] = {}
    for label_id, count in region_counts:
        if label_id == 0:
            continue
        lobe = _label_to_lobe(label_id)
        lobes.setdefault(lobe, [0])
        lobes[lobe][0] += int(count)
    # Convert to dict[lobe -> (voxels, pct)]
    lobes_dict: Dict[str, Dict[str, float]] = {}
    for lobe, (count_list) in lobes.items():
        cnt = count_list if isinstance(count_list, int) else count_list[0]
        lobes_dict[lobe] = {
            "voxels": int(cnt),
            "pct_of_tumor": round(100.0 * cnt / total_voxels, 2),
        }

    # Primary lobe = max
    if lobes_dict:
        primary_lobe = max(lobes_dict, key=lambda L: lobes_dict[L]["voxels"])
        primary_pct = lobes_dict[primary_lobe]["pct_of_tumor"]
    else:
        primary_lobe = "Unknown"
        primary_pct = 0.0

    # Lateralization (in MNI mm space)
    voxel_indices = np.argwhere(mask_arr)  # (N, 3) ijk indices
    coords_mm = _voxels_to_mni_mm(mask_img, voxel_indices)
    left_pct, right_pct = _classify_hemisphere(coords_mm)

    if left_pct > 80:
        lateralization = "left"
    elif right_pct > 80:
        lateralization = "right"
    else:
        lateralization = "bilateral"

    result = {
        "primary_region": primary_lobe,
        "primary_pct": round(primary_pct, 2),
        "regions_involved": regions_involved,
        "lobes": lobes_dict,
        "lateralization": lateralization,
        "left_hemisphere_pct": round(left_pct, 2),
        "right_hemisphere_pct": round(right_pct, 2),
        "total_voxels": total_voxels,
        "total_volume_cm3": round(total_voxels * voxel_volume_mm3 / 1000.0, 3),
    }

    if verbose:
        print(get_anatomical_summary(result))

    return result


# --- Pretty printing --------------------------------------------------------

def get_anatomical_summary(analysis: Dict) -> str:
    """Render an anatomical analysis dict as a clinical-report-style string."""
    lines = []
    lines.append("=" * 64)
    lines.append("ANATOMICAL ANALYSIS")
    lines.append("=" * 64)
    lines.append(f"Total tumor volume    : {analysis['total_volume_cm3']:.2f} cm^3")
    lines.append(f"Total tumor voxels    : {analysis['total_voxels']:,}")
    lines.append("")
    lines.append(f"Primary lobe          : {analysis['primary_region']}  "
                 f"({analysis['primary_pct']:.1f}% of tumor)")
    lines.append(f"Lateralization        : {analysis['lateralization']}")
    lines.append(f"  Left hemisphere     : {analysis['left_hemisphere_pct']:.1f}%")
    lines.append(f"  Right hemisphere    : {analysis['right_hemisphere_pct']:.1f}%")
    lines.append("")
    lines.append("Lobe distribution:")
    for lobe, info in sorted(analysis["lobes"].items(),
                              key=lambda x: -x[1]["voxels"]):
        lines.append(f"  {lobe:<24}: {info['pct_of_tumor']:>5.1f}%  "
                     f"({info['voxels']:,} voxels)")
    lines.append("")
    lines.append("Top anatomical regions involved:")
    for region, voxels, pct in analysis["regions_involved"]:
        lines.append(f"  {region:<48}: {pct:>5.1f}%  ({voxels:,} voxels)")
    lines.append("=" * 64)
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
    analysis = analyze_tumor_anatomy(
        warped_mask=cached["warped_mask"],
        voxel_volume_mm3=1.0,
        verbose=True,
    )



